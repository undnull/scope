#define _USE_MATH_DEFINES 1
#define DR_WAV_IMPLEMENTATION 1

#include <assert.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <math.h>
#include <portaudio.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dr_wav.h"

#define TOSTRING1(x) #x
#define TOSTRING2(x) TOSTRING1(x)

#define BUF_SSBO 0 /* SSBO  - wave data     */
#define BUF_UNIF 1 /* UBO   - common data   */
#define NUM_BUFS 2

typedef double  vec2d_t[2];
typedef float   vec2f_t[2];
typedef float   vec3f_t[3];
typedef float   vec4f_t[4];

struct ubo_data {
    vec4f_t xyz_color;
    vec4f_t x_dt_yz_screen;
};

struct pa_state {
    size_t sample_rate;
    size_t num_samples;
    size_t num_channels;
    size_t position;
    float *samples;
};

static char g_logbuf[4096] = { 0 };
static PaStream *g_stream = NULL;
static GLFWwindow *g_window = NULL;
static GLuint g_program = 0;
static GLuint g_bufs[NUM_BUFS] = { 0 };
static GLuint g_vao = 0;
static struct pa_state g_state = { 0 };
static vec2f_t *g_wave_table = NULL;
static size_t g_wave_table_size = 0;

static const char *vert_src =
    "#version 450 core                                                  \n"
    "layout(binding = 0, std430) buffer __ssbo_0 { vec2 signal[]; };    \n"
    "void main(void)                                                    \n"
    "{                                                                  \n"
    "   gl_Position = vec4(signal[gl_VertexID], 0.0, 1.0);              \n"
    "}                                                                  \n";

static const char *frag_src =
    "#version 450 core                                                  \n"
    "layout(binding = 1, std140) uniform __ubo_1 {                      \n"
    "   vec4 xyz_color;                                                 \n"
    "   vec4 x_dt_yz_screen;                                            \n"
    "};                                                                 \n"
    "layout(location = 0) out vec4 target;                              \n"
    "void main(void)                                                    \n"
    "{                                                                  \n"
    "   target = vec4(xyz_color.xyz, 1.0);                              \n"
    "}";

static void lvprintf(const char *fmt, va_list va)
{
    vsnprintf(g_logbuf, sizeof(g_logbuf), fmt, va);
    fprintf(stderr, "%s\r\n", g_logbuf);
}

static void lprintf(const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    lvprintf(fmt, va);
    va_end(va);
}

static void *safe_malloc(size_t n)
{
    void *block = malloc(n);
    if(!block) {
        lprintf("out of memory!");
        abort();
    }

    return block;
}

static void on_error(int code, const char *message)
{
    lprintf("glfw: %s", message);
}

static GLuint make_shader(GLenum stage, const char *source)
{
    char *info_log;
    GLint status, length;
    GLuint shader;

    shader = glCreateShader(stage);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    if(length > 1) {
        info_log = safe_malloc((size_t)length + 1);
        glGetShaderInfoLog(shader, length, NULL, info_log);
        lprintf("%s", info_log);
        free(info_log);
    }

    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if(!status) {
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

static GLuint make_program(GLuint vert, GLuint frag)
{
    char *info_log;
    GLint status, length;
    GLuint program;

    program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    /* get rid of these */
    glDeleteShader(vert);
    glDeleteShader(frag);

    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    if(length > 1) {
        info_log = safe_malloc((size_t)length + 1);
        glGetProgramInfoLog(program, length, NULL, info_log);
        lprintf("%s", info_log);
        free(info_log);
    }

    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if(!status) {
        glDeleteProgram(program);
        return 0;
    }

    return program;
}

static int pa_callback(const void *input, void *output, unsigned long framerate, const PaStreamCallbackTimeInfo *time_info, PaStreamCallbackFlags flags, void *arg)
{
    size_t j;
    unsigned long i;
    float *fl_output = output;
    struct pa_state *state = arg;
    for(i = 0; i < framerate; i++) {
        if(state->position >= state->num_samples)
            return paComplete;
        for(j = 0; j < state->num_channels; j++)
            *fl_output++ = state->samples[state->position * state->num_channels + j] * 0.25f;
        state->position++;
    }

    return paContinue;
}

static void fill_signal_tab(int scr_width)
{
    int i;
    size_t j, off;
    int64_t scratch = (int64_t)g_state.position - (int64_t)g_state.num_samples;
    size_t num_samples = g_state.num_samples - g_state.position;
    if(num_samples > g_wave_table_size)
        num_samples = g_wave_table_size;
    for(i = 0; i < num_samples; i++) {
        g_wave_table[i][0] = (float)i / (float)num_samples * 2.0f - 1.0f;
        g_wave_table[i][1] = 0.0f;
        off = g_state.position + i;
        if(off >= num_samples) {
            for(j = 0; j < g_state.num_channels; j++)
                g_wave_table[i][1] += g_state.samples[(off - num_samples) * g_state.num_channels + j];
            g_wave_table[i][1] /= (float)g_state.num_channels;
        }
    }
}

static void on_key(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if(action == GLFW_PRESS && key == GLFW_KEY_SPACE && !Pa_IsStreamActive(g_stream))
        Pa_StartStream(g_stream);
}

int main(int argc, char **argv)
{
    drwav wav;
    PaError pa_err;
    PaStreamParameters pa_params;
    GLFWmonitor *monitor;
    const GLFWvidmode *vidmode;
    int width, height;
    double t, pt, dt;
    GLuint vert, frag;
    struct ubo_data ubo;
    size_t width_mod = 1;

    ubo.xyz_color[0] = 1.0f;
    ubo.xyz_color[1] = 1.0f;
    ubo.xyz_color[2] = 1.0f;

    if((pa_err = Pa_Initialize()) != paNoError)
        goto on_pa_error;

    if(argc < 2) {
        lprintf("argument required!");
        return 1;
    }

    if(!drwav_init_file(&wav, argv[1], NULL)) {
        lprintf("unable to open or read %s", argv[1]);
        return 1;
    }

    g_state.sample_rate = wav.sampleRate;
    g_state.samples = safe_malloc(wav.totalPCMFrameCount * wav.channels * sizeof(float));
    g_state.num_samples = drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, g_state.samples);
    g_state.num_channels = wav.channels;
    g_state.position = 0;

    drwav_uninit(&wav);

    if(argc >= 3) {
        width_mod = (size_t)strtoul(argv[2], NULL, 10);
        if(!width_mod)
            width_mod = 1;
    }

    g_wave_table_size = g_state.sample_rate / width_mod;
    g_wave_table = safe_malloc(sizeof(vec2f_t) * g_wave_table_size);

    pa_params.device = Pa_GetDefaultOutputDevice();
    if(pa_params.device == paNoDevice) {
        lprintf("pa: no output device");
        return 1;
    }

    pa_params.channelCount = (int)g_state.num_channels;
    pa_params.sampleFormat = paFloat32;
    pa_params.suggestedLatency = Pa_GetDeviceInfo(pa_params.device)->defaultLowOutputLatency;
    pa_params.hostApiSpecificStreamInfo = NULL;

    pa_err = Pa_OpenStream(&g_stream, NULL, &pa_params, (double)wav.sampleRate, paFramesPerBufferUnspecified, 0, &pa_callback, &g_state);
    if(pa_err != paNoError)
        goto on_pa_error;

    glfwSetErrorCallback(&on_error);

    if(!glfwInit()) {
        lprintf("glfw: init failed");
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    monitor = glfwGetPrimaryMonitor();
    vidmode = glfwGetVideoMode(monitor);
    g_window = glfwCreateWindow(vidmode->width, vidmode->height, "scope", monitor, NULL);
    if(!g_window) {
        lprintf("glfw: window creation failed");
        return 1;
    }

    glfwSetInputMode(g_window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

    glfwMakeContextCurrent(g_window);
    glfwSwapInterval(1);

    if(!gladLoadGL((GLADloadfunc)(&glfwGetProcAddress))) {
        lprintf("glad: loading failed");
        return 1;
    }

    vert = make_shader(GL_VERTEX_SHADER, vert_src);
    frag = make_shader(GL_FRAGMENT_SHADER, frag_src);
    g_program = make_program(vert, frag);
    if(!g_program) {
        lprintf("program compilation failed");
        return 1;
    }

    glCreateBuffers(NUM_BUFS, g_bufs);
    glNamedBufferStorage(g_bufs[BUF_SSBO], sizeof(vec2f_t) * g_wave_table_size, NULL, GL_DYNAMIC_STORAGE_BIT);
    glNamedBufferStorage(g_bufs[BUF_UNIF], sizeof(ubo), NULL, GL_DYNAMIC_STORAGE_BIT);

    /* To draw stuff OpenGL needs a valid VAO
     * bound to the state. We don't need any
     * vertex information because we set things
     * manually or have them hardcoded. */
    glCreateVertexArrays(1, &g_vao);

    glfwSetKeyCallback(g_window, &on_key);

    pt = t = glfwGetTime();
    while(!glfwWindowShouldClose(g_window)) {
        t = glfwGetTime();
        dt = t - pt;
        pt = t;

        glfwGetFramebufferSize(g_window, &width, &height);
        glViewport(0, 0, width, height);
        fill_signal_tab(width);

        ubo.x_dt_yz_screen[0] = (float)dt;
        ubo.x_dt_yz_screen[1] = (float)width;
        ubo.x_dt_yz_screen[2] = (float)height;

        glNamedBufferSubData(g_bufs[BUF_SSBO], 0, sizeof(vec2f_t) * g_wave_table_size, g_wave_table);
        glNamedBufferSubData(g_bufs[BUF_UNIF], 0, sizeof(ubo), &ubo);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, g_bufs[BUF_SSBO]);
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, g_bufs[BUF_UNIF]);

        glBindVertexArray(g_vao);

        glLineWidth(2.0f);

        glUseProgram(g_program);

        glDrawArrays(GL_LINE_STRIP, 0, (GLuint)g_wave_table_size);

        glfwSwapBuffers(g_window);
        glfwPollEvents();
    }

normal_quit:
    glDeleteVertexArrays(1, &g_vao);
    glDeleteBuffers(NUM_BUFS, g_bufs);
    glDeleteProgram(g_program);
    glfwDestroyWindow(g_window);
    glfwTerminate();
    Pa_CloseStream(g_stream);
    free(g_wave_table);
    free(g_state.samples);
    Pa_Terminate();

    return 0;

on_pa_error:
    /* This is a little hacky but it's just a better
     * way to handle this rather than copypasting lots of code. */
    lprintf("PortAudio error: %s", Pa_GetErrorText(pa_err));    
    goto normal_quit;
}

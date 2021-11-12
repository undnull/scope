#define _USE_MATH_DEFINES 1

#include <assert.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOSTRING1(x) #x
#define TOSTRING2(x) TOSTRING1(x)

#define PROG_BEAM 0 /* Signal table (electron beam) */
#define PROG_POST 1 /* Afterimage of the beam       */
#define PROG_GRID 2 /* Oscilloscope's graticule     */
#define NUM_PROGS 3

#define FBO_BEAM 0  /* Actual beam (and occasionally line) image    */
#define FBO_POST 1  /* Afterimage of the beam                       */
#define FBO_COPY 2  /* COPY -> main_fbo and COPY -> POST            */
#define NUM_FBOS 3

#define BUF_SSBO 0 /* SSBO  - signal data */
#define BUF_UNIF 1 /* UBO   - common data */
#define NUM_BUFS 2

#define NDC_MARGIN 0.1

/* This takes about 320 KiB of memory but I think
 * that having a limit at about 22 kHz is worth it. */
#define SIGNAL_TAB_SIZE 40960

#define MAX_SAMPLE_DT (1.0 / 144.0)

typedef double  vec2d_t[2];
typedef float   vec2f_t[2];
typedef float   vec3f_t[3];
typedef float   vec4f_t[4];

struct ubo_data {
    vec4f_t xyz_color;
    vec4f_t x_dt_yz_screen;
};

static char g_logbuf[4096] = { 0 };
static GLFWwindow *g_window = NULL;
static GLuint g_progs[NUM_PROGS] = { 0 };
static GLuint g_fbos_obj[NUM_FBOS] = { 0 };
static GLuint g_fbos_tex[NUM_FBOS] = { 0 };
static GLuint g_bufs[NUM_BUFS] = { 0 };
static GLuint g_vao = 0;
static vec2f_t g_signal[SIGNAL_TAB_SIZE] = { 0 };

static const vec3f_t g_background = { 0.192f, 0.243f, 0.270f };
static const vec3f_t g_foreground = { 0.670f, 0.827f, 0.905f };

/* PROG_BEAM vertex shader.
 * Responsible for translating the long
 * LINE_STRIP or POINTS snake of the signal table. */
static const char *beam_vert_src =
    "#version 450 core                                                                      \n"
    "#define SIGNAL_TAB_SIZE " TOSTRING2(SIGNAL_TAB_SIZE) "                                 \n"
    "#define NDC_MARGIN " TOSTRING2(NDC_MARGIN) "                                           \n"
    "layout(binding = 0, std430) buffer __ssbo_0 { vec2 signal[SIGNAL_TAB_SIZE]; };         \n"
    "void main(void)                                                                        \n"
    "{                                                                                      \n"
    "   uint index = SIGNAL_TAB_SIZE - 1 - gl_VertexID;                                     \n"
    "   gl_Position = vec4(signal[index] * (1.0 - NDC_MARGIN), 0.0, 1.0);                   \n"
    "}                                                                                      \n";

/* PROG_BEAM fragment shader.
 * Responsible for displaying the long
 * LINE_STRIP or POINTS snake of the signal table. */
static const char *beam_frag_src =
    "#version 450 core                                                                      \n"
    "layout(binding = 1, std140) uniform __ubo_1 {                                          \n"
    "   vec4 xyz_color;                                                                     \n"
    "   vec4 x_dt_yz_screen;                                                                \n"
    "};                                                                                     \n"
    "layout(location = 0) out vec4 target;                                                  \n"
    "void main(void)                                                                        \n"
    "{                                                                                      \n"
    "   target = vec4(xyz_color.xyz, 1.0);                                                  \n"
    "}";

/* PROG_POST and PROG_GRID vertex shader.
 * Just draws a full screen-space quad. */
static const char *post_grid_vert_src =
    "#version 450 core                                                                      \n"
    "const vec2 positions[6] = {                                                            \n"
    "   vec2(-1.0, -1.0),                                                                   \n"
    "   vec2(-1.0,  1.0),                                                                   \n"
    "   vec2( 1.0,  1.0),                                                                   \n"
    "   vec2( 1.0,  1.0),                                                                   \n"
    "   vec2( 1.0, -1.0),                                                                   \n"
    "   vec2(-1.0, -1.0),                                                                   \n"
    "};                                                                                     \n"
    "const vec2 texcoords[6] = {                                                            \n"
    "   vec2(0.0, 0.0),                                                                     \n"
    "   vec2(0.0, 1.0),                                                                     \n"
    "   vec2(1.0, 1.0),                                                                     \n"
    "   vec2(1.0, 1.0),                                                                     \n"
    "   vec2(1.0, 0.0),                                                                     \n"
    "   vec2(0.0, 0.0),                                                                     \n"
    "};                                                                                     \n"
    "layout(location = 0) out vec2 texcoord;                                                \n"
    "void main(void)                                                                        \n"
    "{                                                                                      \n"
    "   gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);                               \n"
    "   texcoord = texcoords[gl_VertexID];                                                  \n"
    "}                                                                                      \n";

/* PROG_POST fragment shader.
 * Combines the beam image and the afterimage. */
static const char *post_frag_src =
    "#version 450 core                                                                      \n"
    "layout(binding = 1, std140) uniform __ubo_1 {                                          \n"
    "   vec4 xyz_color;                                                                     \n"
    "   vec4 x_dt_yz_screen;                                                                \n"
    "};                                                                                     \n"
    "layout(location = 0) in vec2 texcoord;                                                 \n"
    "layout(location = 0) out vec4 target;                                                  \n"
    "layout(binding = 0) uniform sampler2D curframe;                                        \n"
    "layout(binding = 1) uniform sampler2D afterimage;                                      \n"
    "vec4 textureBlurCheap(sampler2D s, vec2 b)                                             \n"
    "{                                                                                      \n"
    "   vec2 epsilon = 2.0 / vec2(x_dt_yz_screen.yz);                                       \n"
    "   vec4 res = vec4(0.0);                                                               \n"
    "   res += texture(s, b);                                                               \n"
    "   res += texture(s, b + vec2(epsilon.x, 0.0));                                        \n"
    "   res += texture(s, b - vec2(epsilon.x, 0.0));                                        \n"
    "   res += texture(s, b + vec2(0.0, epsilon.y));                                        \n"
    "   res += texture(s, b - vec2(0.0, epsilon.y));                                        \n"
    "   return res / 5.0;                                                                   \n"
    "}                                                                                      \n"
    "void main(void)                                                                        \n"
    "{                                                                                      \n"
    "   vec4 cc = textureBlurCheap(curframe, texcoord) + texture(curframe, texcoord);       \n"
    "   vec4 ac = textureBlurCheap(afterimage, texcoord) * (1.0 - x_dt_yz_screen.x * 4.0);  \n"
    "   target = max(cc * 0.5, ac);                                                         \n"
    "}                                                                                      \n";

/* PROG_GRID fragment shader.
 * Draws a typical oscilloscope graticule */
static const char *grid_frag_src =
    "#version 450 core                                                                      \n"
    "#define NDC_MARGIN " TOSTRING2(NDC_MARGIN) "                                           \n"
    "layout(binding = 1, std140) uniform __ubo_1 {                                          \n"
    "   vec4 xyz_color;                                                                     \n"
    "   vec4 x_dt_yz_screen;                                                                \n"
    "};                                                                                     \n"
    "layout(location = 0) in vec2 texcoord;                                                 \n"
    "layout(location = 0) out vec4 target;                                                  \n"
    "layout(binding = 0) uniform sampler2D curframe;                                        \n"
    "void main(void)                                                                        \n"
    "{                                                                                      \n"
    "   vec2 tss = x_dt_yz_screen.yz;                                                       \n"
    "   vec2 lim = tss * 0.5 * NDC_MARGIN;                                                  \n"
    "   vec2 oss = tss - 2.0 * lim;                                                         \n"
    "   vec2 cell = oss / 10.0;                                                             \n"
    "   vec2 grid = gl_FragCoord.xy - lim;                                                  \n"
    "   target = texture(curframe, texcoord);                                               \n"
    "   if(grid.x >= 0.0 && grid.y >= 0.0 && grid.x <= oss.x + 1 && grid.y <= oss.y + 1) {  \n"
    "       if(mod(grid.x, cell.x) < 1.0 || mod(grid.y, cell.y) < 1.0) {                    \n"
    "           target *= 0.25;                                                             \n"
    "       }                                                                               \n"
    "   }                                                                                   \n"
    "}                                                                                      \n";

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

static void on_framebuffer_size(GLFWwindow *window, int width, int height)
{
    int i;

    glDeleteTextures(NUM_FBOS, g_fbos_tex);
    glCreateTextures(GL_TEXTURE_2D, NUM_FBOS, g_fbos_tex);

    for(i = 0; i < NUM_FBOS; i++) {
        glTextureStorage2D(g_fbos_tex[i], 1, GL_RGB32F, width, height);
        glTextureParameteri(g_fbos_tex[i], GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(g_fbos_tex[i], GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glNamedFramebufferTexture(g_fbos_obj[i], GL_COLOR_ATTACHMENT0, g_fbos_tex[i], 0);
    }
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

static double make_shm(double a, double t, double f, double phase)
{
    return a * cos(2 * M_PI * t * f + phase);
}

static double make_saw(double a, double t, double f, double phase)
{
    return a * (fmod(t * 2.0 * f + phase, 2.0) - 1.0);
}

static double make_tri(double a, double t, double f, double phase)
{
    return a * asin(cos(2 * M_PI * t * f + phase)) / (0.5 * M_PI);
}

static double make_signal_X(double curtime, double shift)
{
    return make_shm(1.0, curtime, 11000.0, 0.0);
}

static double make_signal_Y(double curtime, double shift)
{
    return make_shm(1.0, curtime, 12000.0, shift);
}

int main(int argc, char **argv)
{
    int i, width, height;
    double xf, t, pt, dt, ts;
    GLuint vert, frag;
    struct ubo_data ubo;

    ubo.xyz_color[0] = g_foreground[0];
    ubo.xyz_color[1] = g_foreground[1];
    ubo.xyz_color[2] = g_foreground[2];

    glfwSetErrorCallback(&on_error);

    if(!glfwInit()) {
        lprintf("glfw: init failed");
        return 1;
    }

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    g_window = glfwCreateWindow(640, 640, "scope", NULL, NULL);
    if(!g_window) {
        lprintf("glfw: window creation failed");
        return 1;
    }

    glfwMakeContextCurrent(g_window);
    glfwSwapInterval(1);

    if(!gladLoadGL((GLADloadfunc)(&glfwGetProcAddress))) {
        lprintf("glad: loading failed");
        return 1;
    }

    /* PROG_BEAM */
    vert = make_shader(GL_VERTEX_SHADER, beam_vert_src);
    frag = make_shader(GL_FRAGMENT_SHADER, beam_frag_src);
    g_progs[PROG_BEAM] = make_program(vert, frag);
    if(!g_progs[PROG_BEAM]) {
        lprintf("prog_beam compilation failed");
        return 1;
    }

    /* PROG_POST */
    vert = make_shader(GL_VERTEX_SHADER, post_grid_vert_src);
    frag = make_shader(GL_FRAGMENT_SHADER, post_frag_src);
    g_progs[PROG_POST] = make_program(vert, frag);
    if(!g_progs[PROG_POST]) {
        lprintf("prot_post compilation failed");
        return 1;
    }

    /* PROG_GRID */
    vert = make_shader(GL_VERTEX_SHADER, post_grid_vert_src);
    frag = make_shader(GL_FRAGMENT_SHADER, grid_frag_src);
    g_progs[PROG_GRID] = make_program(vert, frag);
    if(!g_progs[PROG_GRID]) {
        lprintf("prot_grid compilation failed");
        return 1;
    }

    glCreateFramebuffers(NUM_FBOS, g_fbos_obj);
    glfwGetFramebufferSize(g_window, &width, &height);
    glfwSetFramebufferSizeCallback(g_window, &on_framebuffer_size);
    on_framebuffer_size(g_window, width, height);

    glCreateBuffers(NUM_BUFS, g_bufs);
    glNamedBufferStorage(g_bufs[BUF_SSBO], sizeof(g_signal), NULL, GL_DYNAMIC_STORAGE_BIT);
    glNamedBufferStorage(g_bufs[BUF_UNIF], sizeof(ubo), NULL, GL_DYNAMIC_STORAGE_BIT);

    /* To draw stuff OpenGL needs a valid VAO
     * bound to the state. We don't need any
     * vertex information because we set things
     * manually or have them hardcoded. */
    glCreateVertexArrays(1, &g_vao);

    ts = 0.0;
    pt = t = glfwGetTime();
    while(!glfwWindowShouldClose(g_window)) {
        t = glfwGetTime();
        dt = t - pt;
        pt = t;
        ts += dt;

        for(i = 0; i < SIGNAL_TAB_SIZE; i++) {
            xf = ((double)i / SIGNAL_TAB_SIZE) * dt;
            g_signal[i][0] = (float)make_signal_X(t + xf, ts);
            g_signal[i][1] = (float)make_signal_Y(t + xf, ts);
        }

        glfwGetFramebufferSize(g_window, &width, &height);
        glViewport(0, 0, width, height);

        ubo.x_dt_yz_screen[0] = (float)dt;
        ubo.x_dt_yz_screen[1] = (float)width;
        ubo.x_dt_yz_screen[2] = (float)height;

        glNamedBufferSubData(g_bufs[BUF_SSBO], 0, sizeof(g_signal), g_signal);
        glNamedBufferSubData(g_bufs[BUF_UNIF], 0, sizeof(ubo), &ubo);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, g_bufs[BUF_SSBO]);
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, g_bufs[BUF_UNIF]);

        glBindVertexArray(g_vao);

        /* PROG_BEAM pass */
        glBindFramebuffer(GL_FRAMEBUFFER, g_fbos_obj[FBO_BEAM]);
        glClearColor(g_background[0], g_background[1], g_background[2], 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glLineWidth(4.0);
        glPointSize(4.0);
        glUseProgram(g_progs[PROG_BEAM]);
        glDrawArrays(GL_POINTS, 0, 1);
        glDrawArrays(GL_LINE_STRIP, 0, SIGNAL_TAB_SIZE);

        /* PROG_POST pass */
        glBindFramebuffer(GL_FRAMEBUFFER, g_fbos_obj[FBO_COPY]);
        glUseProgram(g_progs[PROG_POST]);
        glBindTextureUnit(0, g_fbos_tex[FBO_BEAM]);
        glBindTextureUnit(1, g_fbos_tex[FBO_POST]);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBlitNamedFramebuffer(g_fbos_obj[FBO_COPY], g_fbos_obj[FBO_POST], 0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        /* PROG_GRID pass */
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glUseProgram(g_progs[PROG_GRID]);
        glBindTextureUnit(0, g_fbos_tex[FBO_COPY]);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(g_window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &g_vao);
    glDeleteBuffers(NUM_BUFS, g_bufs);
    glDeleteFramebuffers(NUM_FBOS, g_fbos_obj);
    glDeleteTextures(NUM_FBOS, g_fbos_tex);
    glDeleteProgram(g_progs[PROG_GRID]);
    glDeleteProgram(g_progs[PROG_POST]);
    glDeleteProgram(g_progs[PROG_BEAM]);
    glfwDestroyWindow(g_window);
    glfwTerminate();

    return 0;
}

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

#define PROG_MAIN 0 /* draws the dot */
#define PROG_BLIT 1 /* mixes buffers */

#define STAGE_POINT 0
#define STAGE_AFTER 1
#define STAGE_FINAL 2

/* This takes about 160 KiB of memory but I think
 * that having a limit at about 22 kHz is worth it. */
#define SIGNAL_TAB_SIZE 20480

typedef double  vec2d_t[2];
typedef float   vec2f_t[2];
typedef float   vec3f_t[3];

struct framebuffer {
    GLuint fbo;
    GLuint tex;
};

static GLuint programs[2] = { 0 };
static GLuint valid_vao = 0, ssbo = 0;
static struct framebuffer stage_fbos[3] = { 0 };
static vec2f_t signal_tab[SIGNAL_TAB_SIZE] = { 0 };

static const char *vert1_src =
    "#version 450 core                                                              \n"
    "#define SIGNAL_TAB_SIZE " TOSTRING2(SIGNAL_TAB_SIZE) "                         \n"
    "layout(binding = 0, std430) buffer SIGNAL_TAB {                                \n"
    "   vec2 signal[SIGNAL_TAB_SIZE];                                               \n"
    "};                                                                             \n"
    "void main(void)                                                                \n"
    "{                                                                              \n"
    "   uint index = SIGNAL_TAB_SIZE - 1 - gl_VertexID;                             \n"
    "   gl_Position = vec4(signal[index] * 0.90, 0.0, 1.0);                         \n"
    "}                                                                              \n";

static const char *frag1_src =
    "#version 450 core                                                              \n"
    "uniform vec3 color;                                                            \n"
    "layout(location = 0) out vec4 target;                                          \n"
    "void main(void)                                                                \n"
    "{                                                                              \n"
    "   target = vec4(color, 1.0);                                                  \n"
    "}";

static const char *vert2_src =
    "#version 450 core                                                              \n"
    "const vec2 positions[6] = {                                                    \n"
    "   vec2(-1.0, -1.0),                                                           \n"
    "   vec2(-1.0,  1.0),                                                           \n"
    "   vec2( 1.0,  1.0),                                                           \n"
    "   vec2( 1.0,  1.0),                                                           \n"
    "   vec2( 1.0, -1.0),                                                           \n"
    "   vec2(-1.0, -1.0),                                                           \n"
    "};                                                                             \n"
    "const vec2 texcoords[6] = {                                                    \n"
    "   vec2(0.0, 0.0),                                                             \n"
    "   vec2(0.0, 1.0),                                                             \n"
    "   vec2(1.0, 1.0),                                                             \n"
    "   vec2(1.0, 1.0),                                                             \n"
    "   vec2(1.0, 0.0),                                                             \n"
    "   vec2(0.0, 0.0),                                                             \n"
    "};                                                                             \n"
    "layout(location = 0) out vec2 texcoord;                                        \n"
    "void main(void)                                                                \n"
    "{                                                                              \n"
    "   gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);                       \n"
    "   texcoord = texcoords[gl_VertexID];                                          \n"
    "}                                                                              \n";

static const char *frag2_src =
    "#version 450 core                                                              \n"
    "uniform float frametime;                                                       \n"
    "uniform ivec2 screen_size;                                                     \n"
    "layout(location = 0) in vec2 texcoord;                                         \n"
    "layout(location = 0) out vec4 target;                                          \n"
    "layout(binding = 0) uniform sampler2D curframe;                                \n"
    "layout(binding = 1) uniform sampler2D afterburn;                               \n"
    "vec4 textureSmooth(sampler2D s, vec2 b, int n)                                 \n"
    "{                                                                              \n"
    "   vec2 epsilon = 2.0 / vec2(screen_size);                                     \n"
    "   vec4 res = vec4(0.0);                                                       \n"
    "   for(int i = 0; i < n; i++)                                                  \n"
    "       res += texture(s, b);                                                   \n"
    "   res += texture(s, b + vec2(epsilon.x, 0.0));                                \n"
    "   res += texture(s, b - vec2(epsilon.x, 0.0));                                \n"
    "   res += texture(s, b + vec2(0.0, epsilon.y));                                \n"
    "   res += texture(s, b - vec2(0.0, epsilon.y));                                \n"
    "   return res / 5.0;                                                           \n"
    "}                                                                              \n"
    "void main(void)                                                                \n"
    "{                                                                              \n"
    "   vec4 cc = textureSmooth(curframe, texcoord, 3);                             \n"
    "   vec4 ac = textureSmooth(afterburn, texcoord, 1) * (1.0 - frametime * 16.0); \n"
    "   target = max(cc, ac);                                                       \n"
    "}                                                                              \n";

static void die(const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    vfprintf(stderr, fmt, va);
    va_end(va);
    exit(1);
}

static void on_framebuffer_size(GLFWwindow *window, int w, int h)
{
    int i;
    GLuint textures[3] = { stage_fbos[0].tex, stage_fbos[1].tex, stage_fbos[2].tex };
    (void)window;

    glDeleteTextures(3, textures);
    glCreateTextures(GL_TEXTURE_2D, 3, textures);
    for(i = 0; i < 3; i++) {
        glTextureStorage2D(textures[i], 1, GL_RGB32F, w, h);
        glTextureParameteri(textures[i], GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(textures[i], GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glNamedFramebufferTexture(stage_fbos[i].fbo, GL_COLOR_ATTACHMENT0, textures[i], 0);
        stage_fbos[i].tex = textures[i];
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
        info_log = malloc((size_t)length + 1);
        assert((info_log, "Out of memory!"));
        glGetShaderInfoLog(shader, length, NULL, info_log);
        fputs(info_log, stderr);
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
        info_log = malloc((size_t)length + 1);
        assert((info_log, "Out of memory!"));
        glGetProgramInfoLog(program, length, NULL, info_log);
        fputs(info_log, stderr);
        free(info_log);
    }

    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if(!status) {
        glDeleteProgram(program);
        return 0;
    }

    return program;
}

#define PI      (3.14159265359)
#define RATE    (1.0 / 144.0)

static double make_shm(double a, double t, double f, double phase)
{
    return a * cos(2 * PI * t * f + phase);
}

static double make_saw(double a, double t, double f, double phase)
{
    return a * (fmod(t * 2.0 * f + phase, 2.0) - 1.0);
}

static double make_tri(double a, double t, double f, double phase)
{
    return a * asin(cos(2 * PI * t * f + phase)) / (0.5 * PI);
}

static double make_signal_X(double curtime, double phase)
{
    return make_saw(1.0, curtime, 50.0, 0.0);
}

static double make_signal_Y(double curtime, double phase)
{
    return make_shm(0.5, curtime, 150.0, phase);
}

int main(int argc, char **argv)
{
    int i;
    double scratch;
    GLFWwindow *window = NULL;
    int width, height;
    GLuint vert, frag;
    GLuint fbos[3] = { 0 };
    GLint u_color = 0;
    GLint u_frametime = 0;
    GLint u_screen_size = 0;
    const char *glfw_error;
    double curtime, pasttime, frametime, phase, aft;
    vec2f_t *signal_it;
    const vec3f_t dot_color = { 0.324f, 0.926f, 0.684f };

    if(!glfwInit()) {
        glfwGetError(&glfw_error);
        die("GLFW: %s\n", glfw_error);
    }

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    if(!(window = glfwCreateWindow(640, 640, "scope", NULL, NULL))) {
        glfwGetError(&glfw_error);
        die("GLFW: %s\n", glfw_error);
    }

    glfwMakeContextCurrent(window);
    if(!gladLoadGL((GLADloadfunc)(&glfwGetProcAddress)))
        die("gladLoadGL failed\n");
    glfwSwapInterval(0);

    vert = make_shader(GL_VERTEX_SHADER, vert1_src);
    frag = make_shader(GL_FRAGMENT_SHADER, frag1_src);
    assert((vert && frag, "Shader compilation failed"));
    programs[PROG_MAIN] = make_program(vert, frag);
    assert((programs[PROG_MAIN], "PROG_MAIN link failed"));

    /* PROG_MAIN uniforms */
    u_color = glGetUniformLocation(programs[PROG_MAIN], "color");

    vert = make_shader(GL_VERTEX_SHADER, vert2_src);
    frag = make_shader(GL_FRAGMENT_SHADER, frag2_src);
    assert((vert && frag, "Shader compilation failed"));
    programs[PROG_BLIT] = make_program(vert, frag);
    assert((programs[PROG_BLIT], "PROG_BLIT link failed"));

    /* PROG_BLIT uniforms */
    u_frametime = glGetUniformLocation(programs[PROG_BLIT], "frametime");
    u_screen_size = glGetUniformLocation(programs[PROG_BLIT], "screen_size");

    /* To draw stuff OpenGL needs a valid VAO
     * bound to the state. We don't need any
     * vertex information because we set things
     * manually or have them hardcoded. */
    glCreateVertexArrays(1, &valid_vao);

    glCreateBuffers(1, &ssbo);
    glNamedBufferStorage(ssbo, sizeof(signal_tab), NULL, GL_DYNAMIC_STORAGE_BIT);

    glCreateFramebuffers(3, fbos);
    for(i = 0; i < 3; i++)
        stage_fbos[i].fbo = fbos[i];

    glfwGetFramebufferSize(window, &width, &height);
    glfwSetFramebufferSizeCallback(window, &on_framebuffer_size);
    on_framebuffer_size(window, width, height);

    pasttime = curtime = glfwGetTime();
    phase = aft = 0.0f;
    while(!glfwWindowShouldClose(window)) {
        curtime = glfwGetTime();
        frametime = curtime - pasttime;
        pasttime = curtime;
        phase = fmod(phase + frametime, 2.0 * PI);
        aft = (aft + frametime) * 0.5;

        for(i = 0; i < SIGNAL_TAB_SIZE; i++) {
            scratch = ((double)i / (double)SIGNAL_TAB_SIZE) * ((aft < RATE) ? RATE : aft);
            signal_it = signal_tab + i;
            (*signal_it)[0] = (float)make_signal_X(curtime + scratch, phase + scratch);
            (*signal_it)[1] = (float)make_signal_Y(curtime + scratch, phase + scratch);
        }

        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);

        glBindVertexArray(valid_vao);

        glBindFramebuffer(GL_FRAMEBUFFER, stage_fbos[STAGE_POINT].fbo);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glNamedBufferSubData(ssbo, 0, sizeof(signal_tab), signal_tab);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
        glUseProgram(programs[PROG_MAIN]);
        glProgramUniform3fv(programs[PROG_MAIN], u_color, 1, dot_color);
        glLineWidth(4.0f);
        glPointSize(4.0);
        glDrawArrays(GL_POINTS, 0, 1);
        glDrawArrays(GL_LINE_STRIP, 0, SIGNAL_TAB_SIZE);

        glBindFramebuffer(GL_FRAMEBUFFER, stage_fbos[STAGE_FINAL].fbo);
        glUseProgram(programs[PROG_BLIT]);
        glProgramUniform1f(programs[PROG_BLIT], u_frametime, (float)frametime);
        glProgramUniform2i(programs[PROG_BLIT], u_screen_size, width, height);
        glBindTextureUnit(0, stage_fbos[STAGE_POINT].tex);
        glBindTextureUnit(1, stage_fbos[STAGE_AFTER].tex);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glBlitNamedFramebuffer(stage_fbos[STAGE_FINAL].fbo, 0, 0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
        glBlitNamedFramebuffer(stage_fbos[STAGE_FINAL].fbo, stage_fbos[STAGE_AFTER].fbo, 0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    for(i = 0; i < 3; i++)
        fbos[i] = stage_fbos[i].fbo;
    glDeleteFramebuffers(3, fbos);

    /* hack */
    for(i = 0; i < 3; i++)
        fbos[i] = stage_fbos[i].tex;
    glDeleteTextures(3, fbos);

    glDeleteBuffers(1, &ssbo);
    glDeleteVertexArrays(1, &valid_vao);

    glDeleteProgram(programs[PROG_BLIT]);
    glDeleteProgram(programs[PROG_MAIN]);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

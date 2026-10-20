/* Hertzon.exe - starts Hertzon; on first run installs it (Python venv + packages, solver kernels) via setup.ps1.
   Python runs inside this process (python3XY.dll of the venv's base installation), so the Task Manager
   lists the program as Hertzon and not as Python. Should that fail, it starts pythonw.exe as before.
   Build (Linux): x86_64-w64-mingw32-windres launcher.rc -O coff -o launcher.res
                  x86_64-w64-mingw32-gcc -O2 -municode -mwindows -s launcher.c launcher.res -o Hertzon.exe */
#include <windows.h>
#include <shellapi.h>
#include <wchar.h>
#include <string.h>
#include <stdlib.h>

static wchar_t dir[MAX_PATH];

static int exists(const wchar_t *rel) {
    wchar_t p[MAX_PATH * 2];
    swprintf(p, MAX_PATH * 2, L"%ls\\%ls", dir, rel);
    return GetFileAttributesW(p) != INVALID_FILE_ATTRIBUTES;
}

/* first word of a small text file (empty when missing) */
static void first_word(const wchar_t *rel, char *out, int n) {
    wchar_t p[MAX_PATH * 2];
    char buf[256];
    DWORD got = 0;
    int i = 0, j = 0;
    out[0] = 0;
    swprintf(p, MAX_PATH * 2, L"%ls\\%ls", dir, rel);
    HANDLE h = CreateFileW(p, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) return;
    ReadFile(h, buf, sizeof(buf) - 1, &got, NULL);
    CloseHandle(h);
    buf[got] = 0;
    while (buf[i] == ' ' || buf[i] == '\t' || buf[i] == '\r' || buf[i] == '\n') i++;
    while (buf[i] && buf[i] != ' ' && buf[i] != '\t' && buf[i] != '\r' && buf[i] != '\n' && j < n - 1)
        out[j++] = buf[i++];
    out[j] = 0;
}

/* solver kernels compiled for this Hertzon version (marker written by python -m hertzon.solver.warmup) */
static int kernels_current(void) {
    char want[64], have[64];
    first_word(L"hertzon\\solver\\kernels_version.txt", want, sizeof(want));
    first_word(L".venv\\hertzon_kernels.txt", have, sizeof(have));
    return have[0] && strcmp(want, have) == 0;
}

static DWORD run(wchar_t *cmd, DWORD flags, int wait) {
    STARTUPINFOW si = {sizeof(si)};
    PROCESS_INFORMATION pi;
    DWORD code = 0;
    if (!CreateProcessW(NULL, cmd, NULL, NULL, FALSE, flags, NULL, dir, &si, &pi))
        return (DWORD)-1;
    if (wait) {
        WaitForSingleObject(pi.hProcess, INFINITE);
        GetExitCodeProcess(pi.hProcess, &code);
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return code;
}

/* value of "key = value" in .venv\pyvenv.cfg (empty when missing) */
static void venv_cfg(const wchar_t *key, wchar_t *out, int n) {
    wchar_t p[MAX_PATH * 2];
    char buf[4096];
    DWORD got = 0;
    out[0] = 0;
    swprintf(p, MAX_PATH * 2, L"%ls\\.venv\\pyvenv.cfg", dir);
    HANDLE h = CreateFileW(p, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) return;
    ReadFile(h, buf, sizeof(buf) - 1, &got, NULL);
    CloseHandle(h);
    buf[got] = 0;
    char k[64];
    WideCharToMultiByte(CP_UTF8, 0, key, -1, k, sizeof(k), NULL, NULL);
    size_t kl = strlen(k);
    for (char *line = buf; line && *line;) {
        char *next = strchr(line, '\n');
        if (next) *next++ = 0;
        while (*line == ' ' || *line == '\t') line++;
        if (strncmp(line, k, kl) == 0) {
            char *v = line + kl;
            while (*v == ' ' || *v == '\t') v++;
            if (*v == '=') {
                v++;
                while (*v == ' ' || *v == '\t') v++;
                size_t vl = strlen(v);
                while (vl && (v[vl - 1] == '\r' || v[vl - 1] == ' ' || v[vl - 1] == '\t')) v[--vl] = 0;
                MultiByteToWideChar(CP_UTF8, 0, v, -1, out, n);
                return;
            }
        }
        line = next;
    }
}

typedef int (*py_main_t)(int, wchar_t **);

/* run "python -m hertzon ARGS" in this process; 0 when Python could not be loaded */
static int embedded(int *rc) {
    wchar_t home[MAX_PATH], ver[64], dll[MAX_PATH * 2], exe[MAX_PATH * 2];
    venv_cfg(L"home", home, MAX_PATH);
    venv_cfg(L"version", ver, 64);
    if (!home[0] || !exists(L".venv\\Scripts\\pythonw.exe")) return 0;
    int major = 0, minor = 0;
    if (swscanf(ver, L"%d.%d", &major, &minor) != 2) return 0;
    swprintf(dll, MAX_PATH * 2, L"%ls\\python%d%d.dll", home, major, minor);
    /* as the venv's own pythonw.exe does: the base interpreter finds the venv (sys.prefix,
       site-packages, sys.executable) through this variable */
    swprintf(exe, MAX_PATH * 2, L"%ls\\.venv\\Scripts\\pythonw.exe", dir);
    SetEnvironmentVariableW(L"__PYVENV_LAUNCHER__", exe);
    HMODULE py = LoadLibraryExW(dll, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    py_main_t py_main = py ? (py_main_t)GetProcAddress(py, "Py_Main") : NULL;
    if (!py_main) {
        SetEnvironmentVariableW(L"__PYVENV_LAUNCHER__", NULL);
        return 0;
    }
    SetCurrentDirectoryW(dir);              /* as the pythonw.exe process started in this folder */
    int argc = 0;
    wchar_t **argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    wchar_t **pa = (wchar_t **)calloc(argc + 3, sizeof(wchar_t *));
    int k = 0;
    pa[k++] = exe;
    pa[k++] = L"-m";
    pa[k++] = L"hertzon";
    for (int i = 1; argv && i < argc; i++) pa[k++] = argv[i];
    pa[k] = NULL;
    *rc = py_main(k, pa);
    return 1;
}

int WINAPI wWinMain(HINSTANCE h, HINSTANCE p, PWSTR args, int show) {
    wchar_t *cmd;
    size_t n;
    GetModuleFileNameW(NULL, dir, MAX_PATH);
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (slash) *slash = 0;

    if (!exists(L".venv\\hertzon_installed.txt")) {
        if (MessageBoxW(NULL,
                L"Hertzon will now be installed (first start only).\n\n"
                L"This downloads Python packages (EMerge, Qt, VTK, about 1 GB) and takes a few minutes. "
                L"If Python 3.13 is missing it is installed with winget.\n\nContinue?",
                L"Hertzon setup", MB_OKCANCEL | MB_ICONINFORMATION) != IDOK)
            return 0;
        n = wcslen(dir) + 200;
        cmd = (wchar_t *)calloc(n, sizeof(wchar_t));
        swprintf(cmd, n, L"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"%ls\\setup.ps1\"", dir);
        DWORD rc = run(cmd, CREATE_NEW_CONSOLE, 1);
        free(cmd);
        if (rc != 0 || !exists(L".venv\\hertzon_installed.txt")) {
            MessageBoxW(NULL, L"Setup did not finish. See the setup window for the error, then start Hertzon.exe again.",
                        L"Hertzon setup", MB_OK | MB_ICONERROR);
            return 1;
        }
    }
    /* part of the installation: compile the solver kernels with a small test solve, so that the
       program never has to (also once after an update that changed them: kernels_version.txt) */
    if (!kernels_current()) {
        n = wcslen(dir) + 200;
        cmd = (wchar_t *)calloc(n, sizeof(wchar_t));
        swprintf(cmd, n, L"\"%ls\\.venv\\Scripts\\python.exe\" -m hertzon.solver.warmup", dir);
        run(cmd, CREATE_NEW_CONSOLE, 1);
        free(cmd);
    }
    if (!_wgetenv(L"HERTZON_PYTHONW")) {    /* HERTZON_PYTHONW=1: old way, a pythonw.exe process */
        int rc = 0;
        if (embedded(&rc)) return rc;
    }
    n = wcslen(dir) * 2 + wcslen(args) + 100;
    cmd = (wchar_t *)calloc(n, sizeof(wchar_t));
    swprintf(cmd, n, L"\"%ls\\.venv\\Scripts\\pythonw.exe\" -m hertzon %ls", dir, args);
    if (run(cmd, 0, 0) == (DWORD)-1)
        MessageBoxW(NULL, L"Could not start Python. Delete the .venv folder and start Hertzon.exe again to reinstall.",
                    L"Hertzon", MB_OK | MB_ICONERROR);
    free(cmd);
    return 0;
}

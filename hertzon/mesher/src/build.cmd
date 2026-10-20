@echo off
rem Build hzmesh.dll into ..\_bin with MinGW-w64 (gcc/g++ in PATH, e.g. MSYS2 or w64devkit).
rem -ffp-contract=off keeps the exact predicates exact; without it the results differ from Linux.
cd /d "%~dp0"
if not exist build mkdir build
gcc -O2 -ffp-contract=off -c predicates.c -o build\predicates.o || exit /b 1
g++ -O2 -std=c++17 -ffp-contract=off -c hzmesh.cpp -o build\hzmesh.o || exit /b 1
g++ -shared -static -static-libgcc -static-libstdc++ -o build\hzmesh.dll build\hzmesh.o build\predicates.o || exit /b 1
copy /y build\hzmesh.dll ..\_bin\hzmesh.dll

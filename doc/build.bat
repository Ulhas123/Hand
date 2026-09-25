@echo off
rem Build the study handbook PDF.
rem Double-click, or run from a terminal.
rem
rem Two passes are needed: the first writes the table of contents and the page
rem positions used by the title page banner, the second reads them back.
rem
rem XeLaTeX rather than pdfLaTeX: the Python source files contain UTF-8
rem characters (an em dash used as the "no gesture" placeholder), and pdfLaTeX's
rem UTF-8 input encoding makes the `listings` package fail on them with
rem "Invalid UTF-8 byte sequence". XeLaTeX reads UTF-8 natively.
rem
rem latexmk is deliberately NOT used - it requires a Perl runtime that is not
rem part of a standard MiKTeX install.

setlocal
cd /d "%~dp0"

del /q Study-Handbook.aux Study-Handbook.toc Study-Handbook.out 2>nul

xelatex -interaction=nonstopmode Study-Handbook.tex >nul
if errorlevel 1 goto :failed

xelatex -interaction=nonstopmode Study-Handbook.tex >nul
if errorlevel 1 goto :failed

echo.
echo   Built: %~dp0Study-Handbook.pdf
echo.
exit /b 0

:failed
echo.
echo   Build FAILED. Open Study-Handbook.log and search for lines starting "!".
echo.
pause
exit /b 1

echo off
cd /d %~dp0
set root=%userprofile%\anaconda3
call %root%\Scripts\activate.bat %root%
call activate kaolin

:PromptUser
echo off

echo commands:
echo f: fish optimizer
echo s: spline optimizer
echo c: cls

set command=none
set /p command=
if "%command%" == "f" (
	goto RunFishOptimizer
) else if "%command%" == "s" (
	goto RunSplineOptimizer
) else if "%command%" == "c" (
	cls
	goto PromptUser
) else (
	echo skipped unknown command: %command%
	goto EndBatch
)

:RunFishOptimizer
echo on
python "ian_fish_optimizer.py"
goto PromptUser

:RunSplineOptimizer
echo on
python "ian_cubic_spline_optimizer.py"
goto PromptUser

:EndBatch
echo end of batch
cmd /k

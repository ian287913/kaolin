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
echo m: mask segmentation
echo p: pixel filler
echo c: cls
echo e: exit

set command=none
set /p command=
if "%command%" == "f" (
	goto RunFishOptimizer
) else if "%command%" == "s" (
	goto RunSplineOptimizer
) else if "%command%" == "m" (
	goto RunMaskSegmentation
) else if "%command%" == "p" (
	goto RunPixelFiller
) else if "%command%" == "c" (
	cls
	goto PromptUser
) else if "%command%" == "e" (
	goto EndBatch
) else (
	echo skipped unknown command: %command%
	goto PromptUser
)

:RunFishOptimizer
echo on
python "ian_fish_optimizer.py"
goto PromptUser

:RunSplineOptimizer
echo on
python "ian_cubic_spline_optimizer.py"
goto PromptUser

:RunMaskSegmentation
echo target directory:
set target_dir=none
set /p target_dir=
echo on
python "./tools/ian_mask_segmentation.py" "%target_dir%"
goto PromptUser

:RunPixelFiller
echo on
python "ian_pixel_filler.py"
goto PromptUser


:EndBatch
echo end of batch
cmd /k

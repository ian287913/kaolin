echo off
cd /d %~dp0
set root=%userprofile%\anaconda3
call %root%\Scripts\activate.bat %root%
call activate kaolin

:PromptUser
echo off

echo commands:
echo s: segmentation tool
echo c: cls
echo e: exit

set command=none
set /p command=
if "%command%" == "s" (
	goto RunSegmentationTool
) else if "%command%" == "c" (
	cls
	goto PromptUser
) else if "%command%" == "e" (
	goto EndBatch
) else (
	echo skipped unknown command: %command%
	goto PromptUser
)

:RunSegmentationTool
echo on
python "ian_mask_segmentation.py"
goto PromptUser

:EndBatch
echo end of batch
cmd /k

## 專案環境 (Fish Model Generator)
- 作業系統: Windows10 or Windows11
- Fish Model Generator 專案以[kaolin](https://github.com/NVIDIAGameWorks/kaolin)專案為基底，實作mesh生成、mesh optimization與texture optimization。其中主要的class，如renderer、mesh、optimizer是由[Latent-NeRF](https://github.com/eladrich/latent-nerf)專案修改而來。
- kaolin專案環境的安裝方式請參考:
    https://kaolin.readthedocs.io/en/latest/notes/installation.html
    - to check **cuda version**: 
        `$ nvidia-smi`
    - to install pytorch with cuda support:
        https://pytorch.org/get-started/previous-versions/
    - 我使用Anaconda管理python套件環境，"kaolin_environment.yml"檔案內列出了我使用的套件與版本。
- 檔案結構:
```
kaolin (FMG)/                                
├─ examples/                            
│  ├─ tutorial/                                 # FMG 實作，有"ian_"開頭的
│  │  ├─ ian_activate_conda_kaolin.bat          
│  │  ├─ dibr_output/
│  │  ├─ resources/

```
## Repository structure
| Path                                                | Description <img width=200>
|:----------------------------------------------------| :---
| kaolin (FMG)/                                       | Root folder 
| &boxvr;&nbsp; examples/                             | 
| &boxv;&nbsp; &boxvr;&nbsp; tutorial/                | FMG 的實作都在這個資料夾底下，有"ian_"開頭的.py檔案基本上都是FMG會用到的檔案
| &boxv;&nbsp; &boxv;&nbsp; &boxvr;&nbsp; ian_activate_conda_kaolin.bat     | FMG主要功能使用介面
| &boxv;&nbsp; &boxv;&nbsp; &boxvr;&nbsp; dibr_output/      | 存放所有生成結果
| &boxv;&nbsp; &boxv;&nbsp; &boxvr;&nbsp; resources/ | 存放所有生成時所需的資料
| &boxv;&nbsp; &boxv;&nbsp; &boxvr;&nbsp; training    | The `Trainer` class and related code
| &boxv;&nbsp; &boxvr;&nbsp; latent_paint             | Code for `Latent-Paint` training
| &boxv;&nbsp; &boxv;&nbsp; &boxvr;&nbsp; configs     | Config structure for training
| &boxv;&nbsp; &boxv;&nbsp; &boxvr;&nbsp; models      | Textured-Mesh models
| &boxv;&nbsp; &boxv;&nbsp; &boxvr;&nbsp; training    | The `Trainer` class and related code

## ian_activate_conda_kaolin.bat 使用說明
- 執行"ian_activate_conda_kaolin.bat"後，會自動啟動Anaconda中建立的"kaolin"環境，並且提供以下功能選擇:
```
f: fish optimizer
s: spline optimizer
m: mask segmentation
p: pixel filler
c: cls
e: exit
```
- 若輸入`$ f`，則啟動FMG的主要生成功能，此功能會執行"ian_fish_optimizer.py"，
    - "ian_fish_optimizer.py"第84行決定生成所需的所有檔案的母資料夾位置，生成前需要手動修改成想要的資料來源，例如以下路徑將會以"kaolin (FMG)\examples\tutorial\resources\(FREE) Long Fin White Cloud\\"目錄底下的檔案作為input資料:
        ```
        rendered_path_single = "./resources/(FREE) Long Fin White Cloud/"
        ```
    - "ian_fish_optimizer.py"第86行決定生成結果的母資料夾位置，不需要特別修改
        ```
        output_path = './dibr_output/' + str_date_time + '/'
        ```
    - "ian_fish_optimizer.py"中"train_fish()"函式的第216~254行是生成魚身、魚鰭與貼圖的主要迴圈。
    - 第117~120行控制各階段生成的epoch數，例如以下設定將會分配
        - 500 eopch生成魚身；
        - (1200-500) eopch生成魚鰭；
        - (1300-1200) eopch生成貼圖。
        ```
        hyperparameter['num_epoch'] = 1300
        hyperparameter['texture_start_train_epoch'] = 1200
        hyperparameter['fin_start_train_epoch'] = 500
        ```
- 若輸入`$ s`，則啟動hermite spline的optimization測試，基本上不會用到。
- 若輸入`$ m`，則啟動mask segmentation功能
    - 需要接著輸入目標資料夾路徑，例如"C:\Users\ian28\Documents\Github\kaolin\examples\tutorial\resources\(FREE) Long Fin White Cloud"

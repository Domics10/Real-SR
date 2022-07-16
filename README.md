# Changes and configuration made for Tencent/Real-SR to work on HC18 on Docker or Colab

# INDEX

0. Preparazione path e dependencies
1. Preparazione dataset
2. Fase di training
3. Testing
4. Impostazioni per la preparazione alla validazione
5. Avviare la validazione
6. Esecuzione del training con Docker

---

## 0. **Preparazione path e dependencies**
Installare le seguenti librerie necessarie per l'esecuzione dello script:
```bash
!pip install numpy opencv-python lmdb pyyaml
!pip install tb-nightly future
```
Il seguente manuale d'uso è pensato per l'esecuzione del codice Tencent/Real-SR tramite Google Colab, iniziamo con il collegare Google Colab con il proprio Google drive:
```bash
from google.colab import drive
drive.mount('/content/drive')
```
 creare la cartella che conterrà il codice del programma:
```bash
%cd drive/MyDrive && mkdir Tencent && cd Tencent
```
Dopodiché eseguire il download del codice dal seguente github: 
```bash
!git clone https://github.com/Tencent/Real-SR
```
## 1. **Preparazione dataset**

E' necessario scaricare il dataset HC18 :https://zenodo.org/record/1327317#.YpSdTu7P2iM
Provvedere poi al caricamento del dataset su Google Drive. Il dataset contiene delle immagini che esulano dallo scopo del progetto, provvediamo quindi ad eliminarle con il seguente codice:
```bash
!cd ./HC18/training_set && rm *_Annotation*
%cd Real-SR/codes
```
Sarà necessario poi scaricare il github necessario per la creazione dei kernel: procediamo a posizionarci sulla cartella codes dentro Tencent e a scaricare il GitHUb dello script di training:
```bash
!cd preprocess && git clone https://github.com/sefibk/KernelGAN
```
Procediamo alla generazione dei kernel, spostiamoci nella cartella KernelGan:
```bash
!cd ./preprocess/KernelGAN && CUDA_VISIBLE_DEVICES=0 python3 train.py --X4 --input-dir ../HC18/train_set
```
Nella cartella preprocess è necessario modificare il file **paths.yml** e aggiungere la seguente porzione di codice prima della riga datasets e dopo le proprietà del dataset dped:
```yml
hc18:
  clean:
    hr:
      train: '../../HC18/train_set'
      valid: '../../HC18/valid_set'

```
Dopodichè aggiungere tra i path dei datasets esistenti quello di HC18: 
```yml
datasets:
  df2k: '../datasets/DF2K'
  dped: '../datasets/DPED'
  hc18: '../datasets/hc18'
```
Sempre nello stesso path, modificare il file **create_kernel_dataset.py**. In particolare, modificare la riga 63 con il seguente codice:
```python
input_img = Image.open(file).convert('RGB')
```
Posizionarsi poi all'iterno del file **collect_noise.py** e modificare la condizione else a riga 49 con il seguente codice:
```python
if opt.dataset == 'dped':
```
e aggiungere al termine dell'if del dataset riguardante dped la seguente porzione di codice:
```python
else:
        img_dir = PATHS[opt.dataset][opt.artifacts]['hr']['train']
        noise_dir = PATHS['datasets']['hc18'] + '/hc18_noise'
        sp = 256
        max_var = 1000
        min_mean = 50
```
### Creazione dei dataset di training
Eseguire il seguente codice per creare le immagini HR e LR per il dataset selezionato:
```bash
!python3 ./preprocess/create_kernel_dataset.py --dataset hc18 --artifacts clean --kernel_path ./preprocess/KernelGAN/results
```
Dopodichè bisogna inserire il rumore nelle immagini LR create:
```bash
!python3 ./preprocess/collect_noise.py --dataset hc18 --artifacts clean
```
Inserire nella cartella Tencent/Real-SR/codes ed eseguire il comando mkdir pretrained_models, al suo interno, inserire il seguente file: **RRDB_PSNR_x4** disponibile in questo link: https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY
## 2. **Fase di training**

Andare nella cartella Tencent/Real-SR/codes/options/dped e modificare il file **train_kernel_noise.yml**, sostituendo le righe di codice 15, 16 e 17 con il seguente codice:
```yml
noise_data: ../datasets/hc18/hc18_noise/
dataroot_GT: ../datasets/hc18/generated/clean/train_tdsr/HR
dataroot_LQ: ../datasets/hc18/generated/clean/train_tdsr/LR
```
Modificare anche il nome con il nome **HC18_kernel_noise** su riga 2: creerà la cartella con i dati con il nome riportato.
Eseguire in seguito l'allenamento vero e proprio con il comando:
```bash
!CUDA_VISIBLE_DEVICES=0 python3 train.py -opt options/dped/train_kernel_noise.yml
```

### 2.0 Riprendere il traning 
Per riprendere un training interrotto, effettuare le seguente modifiche; andare nella cartella codes/options/deped/train_kernel_noise.yml e modificare la linea 44 come segue :
```yml
resume_state: ../experiments/hc18_kernel_noise/training_state/5000.state
```
A questo punto può essere riavviato il training.

## 3. **Testing**
Modificare il file **test_dped.yml** nella cartella options/dped
modificare la linea 1 così:
```yml
name-- hc18_results
```
linea 13 così:
```yml
dataroot_LR: dataset/hc18/generated/clean/train_tdsr/LR
```
L'allenamento creerà una coppia di files nella cartella experiments/hc18_kernel_noise/models: **5000_D.pth** e **5000_G.pth**, spostare il file **current_step_G.pth**, dove current_step è lo step a cui si è interrotto il training (5000, 10000, 20000 o 30000), nella cartella pretrained_models e sostituire il nome **current_step_G.pth** nella linea linea 26 del file ./codes/option/dped/test_dped.yml, con il nuovo file: 
```yml
pretrained_model_G: ./pretrained_model/current_step_G.pth
```

I risultati finali dell'inferenza saranno generati nella cartella Tencent/Real-SR/results/hc18_results .

Per eseguire il testing eseguire il comando:
```py
CUDA_VISIBLE_DEVICES=1,0 python3 test.py -opt options/dped/test_dped.yml
```

## 4. **Impostazioni per la preparazione alla validazione**

Per consentire al programma di eseguire la fase di validazione sono necessarie delle modifiche ai codici di alcuni file.

Creare una cartella **val_set** nel folder Tencent/HC18 e inserire un numero sufficiente di campioni per eseguire la validazione. Nel nostro caso abbiamo inserito 285 immagini provenienti dal test_set nel val_set.

Modificare il file **create_kernel_dataset.py** nel folder Real-Sr/codes/preprocess, aggiungere i path per le immagini generate attraverso i kernel, inserire il seguente codice da riga 39 all'interno dell'**else**:
```python
path_vsdsr = PATHS['datasets'][opt.dataset] + '/generated/' + opt.artifacts + '/' + "val" + opt.name + '_sdsr/'

path_vtdsr = PATHS['datasets'][opt.dataset] + '/generated/' + opt.artifacts + '/' + "val" + opt.name + '_tdsr/'
```
Inserire il path per le immagini di validazione a riga 48:
```python
input_source_vdir = PATHS[opt.dataset][opt.artifacts]['hr']['val']

input_target_vdir = None
```
Estrarre i source file dalla cartella di validazione, inserire il seguente codice a riga 48-49 sempre dentro **else:
```python
source_files_v = [os.path.join(input_source_vdir, x) for x in os.listdir(input_source_vdir) if utils.is_image_file(x)]
target_files_v = []
```
Creare i path per le immagini che verranno generate dalla validazione, a riga 62 inserire il seguente codice:
```python
if not os.path.exists(vtdsr_hr_dir):

  os.makedirs(vtdsr_hr_dir)

if not os.path.exists(vtdsr_lr_dir):

  os.makedirs(vtdsr_lr_dir)
```
Inserire la seguente funzione con gli argomenti: 
```python
def app_k(source_files, target_files, opt, tdsr_hr_dir, tdsr_lr_dir, kernel_paths, kernel_num):
```
Inseriamo dentro la funzione il blocco di codice già esistente che va da riga 85 a riga 132 
Aggiungere poi al termine del file le seguenti righe:
```python
print("Application to training dataset... ")

app_k(source_files, target_files, opt, tdsr_hr_dir, tdsr_lr_dir, kernel_paths, kernel_num)

print("Application to validation dataset... ")

app_k(source_files_v, target_files_v, opt, vtdsr_hr_dir, vtdsr_lr_dir, kernel_paths, kernel_num)
```
Nella stessa folder di **create_kernel_dataset.py**, modificare il file **collect_noise.py**

Inserire il codice per selezionare il path per create_kernel dataset, inserire il blocco nell'**else** a riga 55:
```py
v_img_dir = PATHS[opt.dataset][opt.artifacts]['hr']['val']
```
Il comando seguente aggiunge una cartella per le immagini a cui è stato aggiunto il rumore:
```py
v_noise_dir = PATHS['datasets']['hc18'] + '/hc18_noise_val'
```
Il seguente comando crea il path, se non esistente, per il rumore, da inserire dopo riga 63:
```py
assert not os.path.exists(v_noise_dir)
    os.mkdir(v_noise_dir)
```
Creare una funzione per il blocco di codice da 73 ad 84 con i seguenti paramentri:
```py
def app_noise(img_dir, noise_dir, sp, max_var, min_mean):
```
Richiamare la funzione in fondo al file, aggiungere il blocco seguente:
```py
print("Collecting noise from : ", img_dir, " to ", noise_dir)
app_noise(img_dir, noise_dir, sp, max_var, min_mean)
print("Collecting noise from : ", v_img_dir, " to ", v_noise_dir)
app_noise(v_img_dir, v_noise_dir, sp, max_var, min_mean)
```
Posizionarsi sul file **train_kernel_noise.yml**con path: Real-SR/codes/option/dped
Aggiungi le seguenti configurazioni:
dentro il tag **dataset** inserire il blocco seguente e riduci**val_freq** a una cifra limitata, nel nostro caso 5000.
```yml
val:
  name: hc18_validation
  mode: LQGT
  aug: noise
  noise_data: ../datasets/hc18/hc18_noise_val/
  dataroot_GT: ../datasets/hc18/generated/clean/val_tdsr/HR
  dataroot_LQ: ../datasets/hc18/generated/clean/val_tdsr/LR
```
## 5. **Avviare il training con validazione**
La validazione viene eseguita con gli stessi comandi visti per il training, si riportano per comodità i comandi da eseguire
```bash
!python3 ./preprocess/create_kernel_dataset.py --dataset hc18 --artifacts clean --kernel_path ./preprocess/KernelGAN/results

!python3 ./preprocess/collect_noise.py --dataset hc18 --artifacts clean

!CUDA_VISIBLE_DEVICES=0 python3 train.py -opt options/dped/train_kernel_noise.yml
```
## 6. **Esecuzione con Docker**

Eseguire le modifiche ai file viste come visto nella sezione Google Colab, dopodiché montare il file **Dockerfile** con questo comando:
```bash
docker build -t nome_immagine .
```
Eseguire poi lo script seguente per lanciare l'esecuzione del container:
```bash
docker run --rm --gpus all --env MODE=modalità  -it -v volume --name  nome_container nome_immagine
```
Le modalità si impostano assegnando specifici valori a MODE:
* kc: creazione del kernel
* ka: applicazione del kernel alle immagini
* na: creazione rumore e combinazione con le imagini
* tr: esecuzione training con validazione
* te: esecuzione testing

A seconda del valore di MODE, cambiano i volumi:
* kc: /mnt/disk1/vrai/CVDL2022/Tencent/Real-SR/codes/preprocess/KernelGAN/results:/home/Tencent/Real-SR/codes/preprocess/KernelGAN/results

* ka: /mnt/disk1/vrai/CVDL2022/Tencent/Real-SR/datasets:/home/Tencent/Real-SR/datasets
* na: /mnt/disk1/vrai/CVDL2022/Tencent/Real-SR/datasets:/home/Tencent/Real-SR/datasets
* tr: /mnt/disk1/vrai/CVDL2022/Tencent/Real-SR/experiments:/home/Tencent/Real-SR/experiments
* te: /mnt/disk1/vrai/CVDL2022/Tencent/Real-SR/codes/results:/home/Tencent/Real-SR/codes/results

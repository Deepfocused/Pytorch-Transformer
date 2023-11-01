>## ***Pytorch Transformer***
* 구현
    * End-to-End Object Detection with Transformers 논문(**DEtection TRansformer** or **DETR**)의 논문을 읽기 전 Transformer 이해가 필수였고, 어쩌다 보니 자연어처리에 관심이 생겨 구현까지 진행(좋은 경험함)
    * Label Smoothing(학습), Beam Search(결과), BLEU(Bilingual Evaluation Understudy) Score 계산(평가) 등은 구현하지 않음

>## ***Development environment***
* OS : ubuntu linux 18.04 LTS
* Graphic card / driver : rtx 2080ti / 418.56
* Anaconda version : 4.10.3
* pytorch version : 1.9.1
    * Configure Run Environment
        1. Create a virtual environment
        ```cmd
        jg@JG:~$ conda create -n pytorch python==3.8.8
        ```
        2. Install the required module 
        ```cmd
        jg@JG:~$ conda activate pytorch 
        (pytorch) jg@JG:~$ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch 
        (pytorch) jg@JG:~$ pip install spacy matplotlib torchtext tensorboard torchsummary torchtext tqdm PyYAML --pre --upgrade
        (pytorch) jg@JG:~$ python -m spacy download en_core_web_sm
        (pytorch) jg@JG:~$ python -m spacy download de_core_news_sm
        ```
>## ***Author*** 

* medical18@naver.com / JONGGON

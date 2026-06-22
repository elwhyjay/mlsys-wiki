> 블로그 출처: https://leimao.github.io/blog/NVIDIA-GPU-Compute-Capability/ , Lei Mao의 글이며 저자의 전재 허가를 받았다. 앞으로 Lei Mao의 CUDA 관련 Blog도 일부 전재할 예정이며, 이는 하나의 완전한 칼럼이다. Blog는 조금 이른 CUDA 아키텍처부터 현재 최신 CUDA 아키텍처까지 다루고, 실용적인 엔지니어링 기법, 하위 수준 명령 분석, Cutlass 분석 등 여러 주제도 포함한다. 시간 흐름이 매우 명확한 칼럼이다.

# NVIDIA GPU Compute Capability

## 소개

서로 다른 NVIDIA GPU의 compute capability(https://developer.nvidia.com/cuda-gpus)를 찾으려면 NVIDIA CUDA GPU web page를 방문할 수 있다. 하지만 NVIDIA GPU가 여러 표에 흩어져 있어 빠르게 검색하기에는 조금 불편하다.

이 블로그 글에서는 모든 NVIDIA GPU와 그 compute capability를 하나의 표로 정리했다. 사용자는 Ctrl + F를 사용해 특정 NVIDIA GPU의 compute capability를 검색할 수 있다.

## NVIDIA GPU Compute Capability
| GPU | 분류 | Compute Capability |
|-----|----------|-------------------|
| NVIDIA Blackwell GPU (GB200) | NVIDIA Data Center Products | 10.0 |
| NVIDIA Blackwell GPU (B200) | NVIDIA Data Center Products | 10.0 |
| GeForce RTX 5090 | GeForce and TITAN Products | 10.0 |
| GeForce RTX 5080 | GeForce and TITAN Products | 10.0 |
| GeForce RTX 5090 | GeForce Notebook Products | 10.0 |
| GeForce RTX 5080 | GeForce Notebook Products | 10.0 |
| NVIDIA H200 | NVIDIA Data Center Products | 9.0 |
| NVIDIA H100 | NVIDIA Data Center Products | 9.0 |
| NVIDIA L4 | NVIDIA Data Center Products | 8.9 |
| NVIDIA L40S | NVIDIA Data Center Products | 8.9 |
| NVIDIA L40 | NVIDIA Data Center Products | 8.9 |
| RTX 6000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 8.9 |
| GeForce RTX 4090 | GeForce and TITAN Products | 8.9 |
| GeForce RTX 4080 | GeForce and TITAN Products | 8.9 |
| GeForce RTX 4070 Ti | GeForce and TITAN Products | 8.9 |
| GeForce RTX 4060 Ti | GeForce and TITAN Products | 8.9 |
| GeForce RTX 4090 | GeForce Notebook Products | 8.9 |
| GeForce RTX 4080 | GeForce Notebook Products | 8.9 |
| GeForce RTX 4070 | GeForce Notebook Products | 8.9 |
| GeForce RTX 4060 | GeForce Notebook Products | 8.9 |
| GeForce RTX 4050 | GeForce Notebook Products | 8.9 |
| Jetson AGX Orin, Jetson Orin NX, Jetson Orin Nano | Jetson Products | 8.7 |
| NVIDIA A40 | NVIDIA Data Center Products | 8.6 |
| NVIDIA A10 | NVIDIA Data Center Products | 8.6 |
| NVIDIA A16 | NVIDIA Data Center Products | 8.6 |
| NVIDIA A2 | NVIDIA Data Center Products | 8.6 |
| RTX A6000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 8.6 |
| RTX A5000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 8.6 |
| RTX A4000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 8.6 |
| RTX A5000 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 8.6 |
| RTX A4000 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 8.6 |
| RTX A3000 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 8.6 |
| RTX A2000 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 8.6 |
| GeForce RTX 3090 Ti | GeForce and TITAN Products | 8.6 |
| GeForce RTX 3090 | GeForce and TITAN Products | 8.6 |
| GeForce RTX 3080 Ti | GeForce and TITAN Products | 8.6 |
| GeForce RTX 3080 | GeForce and TITAN Products | 8.6 |
| GeForce RTX 3070 Ti | GeForce and TITAN Products | 8.6 |
| GeForce RTX 3070 | GeForce and TITAN Products | 8.6 |
| Geforce RTX 3060 Ti | GeForce and TITAN Products | 8.6 |
| Geforce RTX 3060 | GeForce and TITAN Products | 8.6 |
| GeForce RTX 3080 Ti | GeForce Notebook Products | 8.6 |
| GeForce RTX 3080 | GeForce Notebook Products | 8.6 |
| GeForce RTX 3070 Ti | GeForce Notebook Products | 8.6 |
| GeForce RTX 3070 | GeForce Notebook Products | 8.6 |
| GeForce RTX 3060 Ti | GeForce Notebook Products | 8.6 |
| GeForce RTX 3060 | GeForce Notebook Products | 8.6 |
| GeForce RTX 3050 Ti | GeForce Notebook Products | 8.6 |
| GeForce RTX 3050 | GeForce Notebook Products | 8.6 |
| NVIDIA A100 | NVIDIA Data Center Products | 8.0 |
| NVIDIA A30 | NVIDIA Data Center Products | 8.0 |
| NVIDIA T4 | NVIDIA Data Center Products | 7.5 |
| T1000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 7.5 |
| T600 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 7.5 |
| T400 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 7.5 |
| Quadro RTX 8000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 7.5 |
| Quadro RTX 6000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 7.5 |
| Quadro RTX 5000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 7.5 |
| Quadro RTX 4000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 7.5 |
| RTX 5000 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 7.5 |
| RTX 4000 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 7.5 |
| RTX 3000 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 7.5 |
| T2000 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 7.5 |
| T1200 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 7.5 |
| T1000 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 7.5 |
| T600 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 7.5 |
| T500 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 7.5 |
| GeForce GTX 1650 Ti | GeForce and TITAN Products | 7.5 |
| NVIDIA TITAN RTX | GeForce and TITAN Products | 7.5 |
| Geforce RTX 2080 Ti | GeForce and TITAN Products | 7.5 |
| Geforce RTX 2080 | GeForce and TITAN Products | 7.5 |
| Geforce RTX 2070 | GeForce and TITAN Products | 7.5 |
| Geforce RTX 2060 | GeForce and TITAN Products | 7.5 |
| Geforce RTX 2080 | GeForce Notebook Products | 7.5 |
| Geforce RTX 2070 | GeForce Notebook Products | 7.5 |
| Geforce RTX 2060 | GeForce Notebook Products | 7.5 |
| Jetson AGX Xavier, Jetson Xavier NX | Jetson Products | 7.2 |
| NVIDIA V100 | NVIDIA Data Center Products | 7.0 |
| Quadro GV100 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 7.0 |
| NVIDIA TITAN V | GeForce and TITAN Products | 7.0 |
| Jetson TX2 | Jetson Products | 6.2 |
| Tesla P40 | NVIDIA Data Center Products | 6.1 |
| Tesla P4 | NVIDIA Data Center Products | 6.1 |
| Quadro P6000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 6.1 |
| Quadro P5000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 6.1 |
| Quadro P4000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 6.1 |
| Quadro P2200 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 6.1 |
| Quadro P2000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 6.1 |
| Quadro P1000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 6.1 |
| Quadro P620 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 6.1 |
| Quadro P600 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 6.1 |
| Quadro P400 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 6.1 |
| P620 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 6.1 |
| P520 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 6.1 |
| Quadro P5200 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 6.1 |
| Quadro P4200 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 6.1 |
| Quadro P3200 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 6.1 |
| Quadro P5000 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 6.1 |
| Quadro P4000 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 6.1 |
| Quadro P3000 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 6.1 |
| Quadro P2000 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 6.1 |
| Quadro P1000 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 6.1 |
| Quadro P600 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 6.1 |
| Quadro P500 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 6.1 |
| NVIDIA TITAN Xp | GeForce and TITAN Products | 6.1 |
| NVIDIA TITAN X | GeForce and TITAN Products | 6.1 |
| GeForce GTX 1080 Ti | GeForce and TITAN Products | 6.1 |
| GeForce GTX 1080 | GeForce and TITAN Products | 6.1 |
| GeForce GTX 1070 Ti | GeForce and TITAN Products | 6.1 |
| GeForce GTX 1070 | GeForce and TITAN Products | 6.1 |
| GeForce GTX 1060 | GeForce and TITAN Products | 6.1 |
| GeForce GTX 1050 | GeForce and TITAN Products | 6.1 |
| GeForce GTX 1080 | GeForce Notebook Products | 6.1 |
| GeForce GTX 1070 | GeForce Notebook Products | 6.1 |
| GeForce GTX 1060 | GeForce Notebook Products | 6.1 |
| Tesla P100 | NVIDIA Data Center Products | 6.0 |
| Quadro GP100 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 6.0 |
| Jetson Nano | Jetson Products | 5.3 |
| Tesla M60 | NVIDIA Data Center Products | 5.2 |
| Tesla M40 | NVIDIA Data Center Products | 5.2 |
| Quadro M6000 24GB | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 5.2 |
| Quadro M6000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 5.2 |
| Quadro M5000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 5.2 |
| Quadro M4000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 5.2 |
| Quadro M2000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 5.2 |
| Quadro M5500M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 5.2 |
| Quadro M2200 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 5.2 |
| Quadro M620 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 5.2 |
| GeForce GTX TITAN X | GeForce and TITAN Products | 5.2 |
| GeForce GTX 980 Ti | GeForce and TITAN Products | 5.2 |
| GeForce GTX 980 | GeForce and TITAN Products | 5.2 |
| GeForce GTX 970 | GeForce and TITAN Products | 5.2 |
| GeForce GTX 960 | GeForce and TITAN Products | 5.2 |
| GeForce GTX 950 | GeForce and TITAN Products | 5.2 |
| GeForce GTX 980 | GeForce Notebook Products | 5.2 |
| GeForce GTX 980M | GeForce Notebook Products | 5.2 |
| GeForce GTX 970M | GeForce Notebook Products | 5.2 |
| GeForce GTX 965M | GeForce Notebook Products | 5.2 |
| GeForce 910M | GeForce Notebook Products | 5.2 |
| Quadro K2200 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 5.0 |
| Quadro K1200 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 5.0 |
| Quadro K620 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 5.0 |
| Quadro M1200 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 5.0 |
| Quadro M520 | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 5.0 |
| Quadro M5000M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 5.0 |
| Quadro M4000M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 5.0 |
| Quadro M3000M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 5.0 |
| Quadro M2000M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 5.0 |
| Quadro M1000M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 5.0 |
| Quadro K620M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 5.0 |
| Quadro M600M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 5.0 |
| Quadro M500M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 5.0 |
| NVIDIA NVS 810 | Desktop Products | 5.0 |
| GeForce GTX 750 Ti | GeForce and TITAN Products | 5.0 |
| GeForce GTX 750 | GeForce and TITAN Products | 5.0 |
| GeForce GTX 960M | GeForce Notebook Products | 5.0 |
| GeForce GTX 950M | GeForce Notebook Products | 5.0 |
| GeForce 940M | GeForce Notebook Products | 5.0 |
| GeForce 930M | GeForce Notebook Products | 5.0 |
| GeForce GTX 850M | GeForce Notebook Products | 5.0 |
| GeForce 840M | GeForce Notebook Products | 5.0 |
| GeForce 830M | GeForce Notebook Products | 5.0 |
| Tesla K80 | Tesla Workstation Products | 3.7 |
| Tesla K80 | NVIDIA Data Center Products | 3.7 |
| Tesla K40 | Tesla Workstation Products | 3.5 |
| Tesla K20 | Tesla Workstation Products | 3.5 |
| Tesla K40 | NVIDIA Data Center Products | 3.5 |
| Tesla K20 | NVIDIA Data Center Products | 3.5 |
| Quadro K6000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 3.5 |
| Quadro K5200 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 3.5 |
| Quadro K610M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 3.5 |
| Quadro K510M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 3.5 |
| GeForce GTX TITAN Z | GeForce and TITAN Products | 3.5 |
| GeForce GTX TITAN Black | GeForce and TITAN Products | 3.5 |
| GeForce GTX TITAN | GeForce and TITAN Products | 3.5 |
| GeForce GTX 780 Ti | GeForce and TITAN Products | 3.5 |
| GeForce GTX 780 | GeForce and TITAN Products | 3.5 |
| GeForce GT 730 | GeForce and TITAN Products | 3.5 |
| GeForce GT 720 | GeForce and TITAN Products | 3.5 |
| GeForce GT 705* | GeForce and TITAN Products | 3.5 |
| GeForce GT 640 (GDDR5) | GeForce and TITAN Products | 3.5 |
| GeForce 920M | GeForce Notebook Products | 3.5 |
| Tesla K10 | NVIDIA Data Center Products | 3.0 |
| Quadro K5000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 3.0 |
| Quadro K4200 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 3.0 |
| Quadro K4000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 3.0 |
| Quadro K2000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 3.0 |
| Quadro K2000D | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 3.0 |
| Quadro K600 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 3.0 |
| Quadro K420 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 3.0 |
| Quadro 410 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 3.0 |
| Quadro K6000M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 3.0 |
| Quadro K5200M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 3.0 |
| Quadro K5100M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 3.0 |
| Quadro K500M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 3.0 |
| Quadro K4200M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 3.0 |
| Quadro K4100M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 3.0 |
| Quadro K3100M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 3.0 |
| Quadro K2200M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 3.0 |
| Quadro K2100M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 3.0 |
| Quadro K1100M | NVIDIA Quadro and NVIDIA RTX Mobile GPUs | 3.0 |
| NVIDIA NVS 510 | Desktop Products | 3.0 |
| GeForce GTX 770 | GeForce and TITAN Products | 3.0 |
| GeForce GTX 760 | GeForce and TITAN Products | 3.0 |
| GeForce GTX 690 | GeForce and TITAN Products | 3.0 |
| GeForce GTX 680 | GeForce and TITAN Products | 3.0 |
| GeForce GTX 670 | GeForce and TITAN Products | 3.0 |
| GeForce GTX 660 Ti | GeForce and TITAN Products | 3.0 |
| GeForce GTX 660 | GeForce and TITAN Products | 3.0 |
| GeForce GTX 650 Ti BOOST | GeForce and TITAN Products | 3.0 |
| GeForce GTX 650 Ti | GeForce and TITAN Products | 3.0 |
| GeForce GTX 650 | GeForce and TITAN Products | 3.0 |
| GeForce GT 740 | GeForce and TITAN Products | 3.0 |
| GeForce GTX 880M | GeForce Notebook Products | 3.0 |
| GeForce GTX 870M | GeForce Notebook Products | 3.0 |
| GeForce GTX 780M | GeForce Notebook Products | 3.0 |
| GeForce GTX 770M | GeForce Notebook Products | 3.0 |
| GeForce GTX 765M | GeForce Notebook Products | 3.0 |
| GeForce GTX 760M | GeForce Notebook Products | 3.0 |
| GeForce GTX 680MX | GeForce Notebook Products | 3.0 |
| GeForce GTX 680M | GeForce Notebook Products | 3.0 |
| GeForce GTX 675MX | GeForce Notebook Products | 3.0 |
| GeForce GTX 670MX | GeForce Notebook Products | 3.0 |
| GeForce GTX 660M | GeForce Notebook Products | 3.0 |
| GeForce GT 755M | GeForce Notebook Products | 3.0 |
| GeForce GT 750M | GeForce Notebook Products | 3.0 |
| GeForce GT 650M | GeForce Notebook Products | 3.0 |
| GeForce GT 745M | GeForce Notebook Products | 3.0 |
| GeForce GT 645M | GeForce Notebook Products | 3.0 |
| GeForce GT 740M | GeForce Notebook Products | 3.0 |
| GeForce GT 730M | GeForce Notebook Products | 3.0 |
| GeForce GT 640M | GeForce Notebook Products | 3.0 |
| GeForce GT 640M LE | GeForce Notebook Products | 3.0 |
| GeForce GT 735M | GeForce Notebook Products | 3.0 |
| GeForce GT 730M | GeForce Notebook Products | 3.0 |
| NVIDIA NVS 315 | Desktop Products | 2.1 |
| NVIDIA NVS 310 | Desktop Products | 2.1 |
| NVS 5400M | Mobile Products | 2.1 |
| NVS 5200M | Mobile Products | 2.1 |
| NVS 4200M | Mobile Products | 2.1 |
| GeForce GTX 560 Ti | GeForce and TITAN Products | 2.1 |
| GeForce GTX 550 Ti | GeForce and TITAN Products | 2.1 |
| GeForce GTX 460 | GeForce and TITAN Products | 2.1 |
| GeForce GTS 450 | GeForce and TITAN Products | 2.1 |
| GeForce GTS 450* | GeForce and TITAN Products | 2.1 |
| GeForce GT 730 DDR3,128bit | GeForce and TITAN Products | 2.1 |
| GeForce GT 640 (GDDR3) | GeForce and TITAN Products | 2.1 |
| GeForce GT 630 | GeForce and TITAN Products | 2.1 |
| GeForce GT 620 | GeForce and TITAN Products | 2.1 |
| GeForce GT 610 | GeForce and TITAN Products | 2.1 |
| GeForce GT 520 | GeForce and TITAN Products | 2.1 |
| GeForce GT 440 | GeForce and TITAN Products | 2.1 |
| GeForce GT 440* | GeForce and TITAN Products | 2.1 |
| GeForce GT 430 | GeForce and TITAN Products | 2.1 |
| GeForce GT 430* | GeForce and TITAN Products | 2.1 |
| GeForce 820M | GeForce Notebook Products | 2.1 |
| GeForce 800M | GeForce Notebook Products | 2.1 |
| GeForce GTX 675M | GeForce Notebook Products | 2.1 |
| GeForce GTX 670M | GeForce Notebook Products | 2.1 |
| GeForce GT 635M | GeForce Notebook Products | 2.1 |
| GeForce GT 630M | GeForce Notebook Products | 2.1 |
| GeForce GT 625M | GeForce Notebook Products | 2.1 |
| GeForce GT 720M | GeForce Notebook Products | 2.1 |
| GeForce GT 620M | GeForce Notebook Products | 2.1 |
| GeForce 710M | GeForce Notebook Products | 2.1 |
| GeForce 705M | GeForce Notebook Products | 2.1 |
| GeForce 610M | GeForce Notebook Products | 2.1 |
| GeForce GTX 580M | GeForce Notebook Products | 2.1 |
| GeForce GTX 570M | GeForce Notebook Products | 2.1 |
| GeForce GTX 560M | GeForce Notebook Products | 2.1 |
| GeForce GT 555M | GeForce Notebook Products | 2.1 |
| GeForce GT 550M | GeForce Notebook Products | 2.1 |
| GeForce GT 540M | GeForce Notebook Products | 2.1 |
| GeForce GT 525M | GeForce Notebook Products | 2.1 |
| GeForce GT 520MX | GeForce Notebook Products | 2.1 |
| GeForce GT 520M | GeForce Notebook Products | 2.1 |
| GeForce GTX 485M | GeForce Notebook Products | 2.1 |
| GeForce GTX 470M | GeForce Notebook Products | 2.1 |
| GeForce GTX 460M | GeForce Notebook Products | 2.1 |
| GeForce GT 445M | GeForce Notebook Products | 2.1 |
| GeForce GT 435M | GeForce Notebook Products | 2.1 |
| GeForce GT 420M | GeForce Notebook Products | 2.1 |
| GeForce GT 415M | GeForce Notebook Products | 2.1 |
| GeForce 710M | GeForce Notebook Products | 2.1 |
| GeForce 410M | GeForce Notebook Products | 2.1 |
| Tesla C2075 | Tesla Workstation Products | 2.0 |
| Tesla C2050/C2070 | Tesla Workstation Products | 2.0 |
| Quadro Plex 7000 | NVIDIA Quadro and NVIDIA RTX Desktop GPUs | 2.0 |
| GeForce GTX 590 | GeForce and TITAN Products | 2.0 |
| GeForce GTX 580 | GeForce and TITAN Products | 2.0 |
| GeForce GTX 570 | GeForce and TITAN Products | 2.0 |
| GeForce GTX 480 | GeForce and TITAN Products | 2.0 |
| GeForce GTX 470 | GeForce and TITAN Products | 2.0 |
| GeForce GTX 465 | GeForce and TITAN Products | 2.0 |
| GeForce GTX 480M | GeForce Notebook Products | 2.0 |
| GeForce GTX 860M | GeForce Notebook Products | 3.0/5.0(**) |

## 기타 정보

위 표를 생성하는 데 사용한 Python script는 내 Gist(https://gist.github.com/leimao/5c5cffc01f0db8b4334ace3267ddc851)에서 찾을 수 있다.

## 참고 자료

- Compute Capability - CUDA Programming Guide(https://docs.nvidia.com/cuda/archive/12.6.2/cuda-c-programming-guide/index.html#compute-capabilities)
- NVIDIA GPU Compute Capability(https://developer.nvidia.com/cuda-gpus)

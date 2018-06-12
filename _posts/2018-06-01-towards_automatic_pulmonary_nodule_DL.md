---
header: Towards automatic pulmonary nodule management in lung cancer screening with deep learning
mathjax: True
---
<!--more-->

기존의 feature를 뽑아서 classic ML 방식으로 폐 결절(nodule)의 class를 분류 문제를 딥러닝의 convolution network를 사용한 새로운 구조(multi-stream multi-scale)를 single system으로 분류하는 방법을 처음으로 제시했습니다. 이 때, 성능이 classic ML보다 나으며 사람 평가 성능과 유사한 수준임을 보입니다.

<!--more-->

***
### 폐암에 관한 배경지식
14년에 발표한 2012년 암 통계 자료에 따르면, 폐암 발생률은 남자는 전체 암의 13.7%를 차지하며 위암 대장암에 이어 세 번째이고, 여자는 6.0%를 차지하며 갑상선암, 유방암, 대장암, 위암에 이어 다섯 번째이다. 그러나 암사망은 남자의 경우 폐암이 전체 암 사망의 26.6%를 차지하고 여자의 경우 16.5%를 차지하여 남녀 모두 암종별 사망률 1위입니다. 이처럼 높은 사망률의 원인은 다른 암과 비교했을 때 같은 병기별 생존율도 낮을 뿐 아니라 처음 진단 시 높은 병기가 많기 때문입니다.[1] 따라서 가능한 이른 병기의 폐암을 발견하는 것이 폐암환자의 생존율 뿐 아니라 의료비도 감소시킬 수 있습니다.
<figure>
	<img src="/img/1/lung_cancer_death_ratio.png" alt="alt text">
	<figcaption>그림0. 주요 암종별 사망률</figcaption>
</figure>	
미국에서는 NCI (National Cancer Institute) 의 주도 하에 NLST (National Lung Screening Trial)[[2]](https://www.nejm.org/doi/full/10.1056/nejmoa1102873) 을 진행한 바 있습니다. 2002년부터 약 2년간 총 53,454명의 흡연력이 있는 55 - 74세의 일반인에 대해 저선량 CT로 Lung screening을 진행하였고 그 결과 기존 X-ray 검진 대비  15~20% 사망률을 줄일 수 있다는 결과를 얻었습니다.  
NLST의 성공적인 결과로 2004년부터 미국 USPSTF (U.S. Preventive Service Task Force) 에서는 30갑년 흡연력을 가진 55세 - 80세 고위험군을 대상으로 매년 저선량 CT를 통해 스크리닝 테스트를 받을 것을 권장하고 있습니다. 
대한민국에서도 폐암 검진 시범 서비스를 2017년부터 2018년까지 2년에 걸쳐 약 8,000명의 고위험군 대상으로 진행하였으며[[3]](http://www.mohw.go.kr/react/al/sal0301vw.jsp?PAR_MENU_ID=04&MENU_ID=0403&CONT_SEQ=342899&page=1), 2019년부터는 6대암에 편입하여 만 55∼74세 흡연 고위험군을 대상으로 무료 검진이 가능하도록 한다고 합니다. 

### 폐 결절(nodule) 분류 
폐 결절은 아래 표와 같이 크게 3가지, 세분화되어 5가지 class로 분류 할 수 있습니다. 이렇듯 결절의 종류를 나눠놓은 이유는 종류에 따라 **악성 가능성**을 판단 할 수 있기 때문입니다. 
병원에서는 방사선의가 아래 표와 같은 CT 이미지를 보고 결절를 분류합니다. 분명한 분류 기준[[4]](https://www.acr.org/-/media/ACR/Files/RADS/Lung-RADS/LungRADS_AssessmentCategories.pdf?la=en) 을 토대로 판단하지만 주관이 완벽하게 배제될 수는 없기 때문에 분류하는 사람마다 어느 정도 편차가 생기게 됩니다. 그렇기 때문에 이를 고려하는 평가 기준이 필요합니다. 
<figure>
	<img src="/img/1/nodule_classes.png" alt="alt text">
	<figcaption>그림1. 폐 결절 분류</figcaption>
</figure>	

### 평가 기준(Inter-observer variability)
사람마다의 편차로 인해 분류 결과가 다를 수 있다는 것을 앞서 살펴봤습니다. 이 논문에서는 이러한 편차를 고려한 정확도를 고려하기 위해서 서로 다른 평가자의 동의 정도를 나타내는 통계량인 Cohen k statistics[[5]](https://en.wikipedia.org/wiki/Cohen%27s_kappa) 를 도입해서 의사 vs 의사, 의사 vs 컴퓨터 간의 성능을 측정합니다. 궁극적으로 **논문에서 보이고자 하는 내용은 의사, 컴퓨터 간의 평가 결과가 의사들 간의 결과 못지 않다는 것입니다.**
- Cohen k statistics: $$\kappa \equiv \dfrac {p_o - p_e} {1 - p_e} = 1 - \dfrac {1 - p_o} {1 - p_e}$$ <br>
($$p_o$$: the observed proportionate agreement(accuracy), $$p_e$$: the probability of random agreement) 

### Multi-stream multi-scale 구조
이미지를 잘 다루는 CNN(Convolutional Neural Network)를 기본 구조로 사용한 분류 문제로 접근합니다.그리고 결절의 특징이 잘 학습될 수 있도록 입력 이미지에 분류에 필요한 크기(scale), 연속성(stream) 2가지를 고려해줍니다.

- Scale 
	- perifissural은 결절 주위 림프절의 존재 여부로 분류합니다. 결절의 크기를 조절하면서 관찰했을 때 축소(x1)된 경우에만 림프절이 관찰되어 perifissural로 분류되지만, 확대(x4)했을 때는 관찰되지 않아 다른 종류의 결절로 분류될 가능성이 있어 보입니다.
	<figure>
		<img src="/img/1/perifissural.png" alt="alt text" title="Title Text">
  		<figcaption>그림2. perifissural 결절 이미지 (크기 순서대로 x4, x2, x1) </figcaption>
	</figure>
	- non-solid와 part-solid는 solid core의 유무로 분류합니다. 결절의 크기를 조절하면서 관찰했을 때 축소(x1)된 경우에는 solid core가 잘 구분되지 않아 분류가 어렵지만, 확대(x4)했을 때는 명확하게 part-solid를 구분 할 수 있습니다.
	<figure>
		<img src="/img/1/non_solid_part_solid.png" alt="alt text" title="Title Text">
  		<figcaption>그림3. non-solid 결절(1행), part-solid 결절(2행) </figcaption>
	</figure>	
- Stream
	- CT를 찍으면 3-D로 구성된 이미지 정보를 얻을 수 있습니다. Multi-stream은 이 정보들을 최대한 다양하게 활용하자는 접근 방법입니다. axial, coronal, sagittal 3개의 평면 축을 기준으로 일정 각도만큼 돌렸을 때 생성되는 단면을 입력 데이터로 사용합니다. 이 과정을 통해 단순 axial에 대한 학습을 했을 때보다 더 다양한 결절 형태를 고려 할 수 있게 됩니다. 
	<figure>
		<img src="/img/1/augmentation.png" alt="alt text" title="Title Text">
  		<figcaption>그림4. N에 따른 multi-stream 데이터 augmenatation 시각화</figcaption>
	</figure>	

### Methods
- 데이터
	- Multi-Scale: 결절의 영역을 크기(d = 10, 20, 40mm)별 정사각형으로 잘라내서 resize 해줍니다. 
	- Multi-Stream: 3개 평면에 각도 변화를 줬을 때 스캔과 만나는 단면을 데이터로 사용합니다.  
	- Prediction: 학습 데이터를 각도에 따라서 생성해서 사용했기 때문에 결과도 각도에 dependent하게 됩니다. 따라서 예측할 때에도 각도를 고려해서($N=30$) 생성된 이미지에 대한 예측 확률의 평균으로 계산합니다.($$y=\arg\max_k(\frac 1 N \Sigma_{i=1}^N P_k(\mathbf{x}_{\theta})$$)
- 모델 구조
	- scale 개수(d) 만큼의 2-D patch를 VGG 기반 모델에 입력 값으로 넣어주며 각각 얻은 feature들을 합쳐서 Fully Connected Layer에서 각 class에 대한 확률을 예측을 합니다. 이 때, 같은 scale에 대한 network 간 parameter는 공유됩니다. 
	<figure>
		<img src="/img/1/model.png" alt="alt text" title="Title Text">
  		<figcaption>그림5. 모델 구조</figcaption>
	</figure>	

### Datasets
데이터는 class에 따라 나누고 augmentation을 통해 대략 class별 80,000장으로 맞춰줍니다. test 데이터는 $test_{ALL}$, $test_{OBS}$로 두 가지로 나눠서 평가합니다.
$test_{OBS}$는 각 class 별 개수를 동일하게 맞춰서 만든 데이터셋이고 $test_{ALL}$은 비율에 관계없이 모든 test 데이터셋을 포함합니다.
<figure>
	<img src="/img/1/dataset.png" alt="alt text" title="Title Text">
	<figcaption>그림6. 데이터셋 구성</figcaption>
</figure>	

### Observer study
의사, 컴퓨터 간의 Inter-observer variation, F-Score를 구해서 비교합니다. $$test_{OBS}$$ 데이터를 활용하고 평가에 참여하는 의사는 3명($$O_1, O_2, O_3$$) 으로 구성되며 모두가 20년 이상의 경력을 가진 radiology researcher입니다. 그리고 $$O_4$$는 데이터셋에(DLCST) 주어진 annotation으로 설정합니다. 컴퓨터는 scale 별로 학습시킨 모델을 사용합니다.
- 표1: 의사-의사는 0.59-0.75, 의사-컴퓨터는 성능이 가장 좋은 3-scale에 한정시키면 0.58-0.67로 의사 간 비교 결과의 범위 내에 존재하며 일부 의사-의사 결과보다 더 높은 값을 내는 것으로 보입니다. scale이 많아질 수록 성능이 높아지므로 Multi-scale에 대한 유효성을 실험적으로 보입니다.
<figure>
	<img src="/img/1/eval_result2.png" alt="alt text" title="Title Text">
	<figcaption>표1. 의사-의사, 의사-컴퓨터 간 inter-observer variation 결과 </figcaption>
</figure>	
- 표2: 의사(정답)-의사, 의사(정답)-컴퓨터 간 F-score를 구한 결과로 평균적으로 72.9%, 69.6%의 유사한 결과를 나타냅니다.
<figure>
	<img src="/img/1/eval_result1.png" alt="alt text" title="Title Text">
	<figcaption>표2. 의사-의사, 의사-컴퓨터 간 F-Score 결과</figcaption>
</figure>	


### Discussion
- 타 ML 알고리즘과 비교: 모든 class에 대해서 딥러닝이 잘 예측합니다. Observer study 결과와 비교했을 때 spiculated, part-solid 성능이 크게 낮아졌습니다. 원인은 데이터가 적어서 유사한 class 간의 특징을 모델이 충분히 학습하지 못한 것으로 보이며 다양한 데이터를 추가해 줄 경우 성능이 올라갈 것으로 보입니다. 
<figure>
	<img src="/img/1/comparison_to_ML.png" alt="alt text" title="Title Text">
	<figcaption>표3. SVM - ConvNet 성능 비교</figcaption>
</figure>	

- t-SNE 분석: t-SNE는 feature 간의 유사도를 시각화 할 때 사용하는 방법입니다. 모델의 최종 feature를 시각화했을 때 아래와 같은 이미지의 군집을 얻을 수 있습니다. 보시면, large solid 와 spiculated이 같은 군집 내에 위치한 것을 볼 수 있고 모델이 둘 간의 구분을 어려워 할 것을 예상 할 수 있습니다. (표3에서 spiculated의 성능이 낮게 나온 이유가 large solid의 FP로 추정이 가능합니다.)
<figure>
	<img src="/img/1/tsne_result.png" alt="alt text" title="Title Text">
	<figcaption>그림7. 결절 class 별 t-SNE 분포</figcaption>
</figure>	
- 예측 결과 분석: 예측된 결절의 형태가 전형적인지, 비전형적인지에 따라 예측 확률을 살펴봅니다. 사람이 라벨링을 한 데이터를 학습시켰기 때문에 전형적인(많은) 데이터에 대해서는 높은 확률로 예측을 잘 하고, 비전형적인(상대적으로 적은) 데이터에 대해서는 낮은 확률로 예측을 하는 것을 확인 할 수 있습니다. 
<figure>
	<img src="/img/1/pred_analysis.png" alt="alt text" title="Title Text">
	<figcaption>그림8. 결절의 형태에 따른 예측 결과</figcaption>
</figure>	


### Reference
[1] [Kyu-Won Jung. Cancer Statistics in Korea: Incidence, Mortality, Survival and Prevalence in 2015](https://www.e-crt.org/journal/view.php?number=2850) <br>
[2] [NLST research team. Reduced Lung-Cancer Mortality with Low-Dose Computed Tomographic Screening. N Engl J. 2011](https://www.nejm.org/doi/full/10.1056/nejmoa1102873)<br>
[3] [대한민국 폐암검진 시범 사업. 보건복지부](http://www.mohw.go.kr/react/al/sal0301vw.jsp?PAR_MENU_ID=04&MENU_ID=0403&CONT_SEQ=342899&page=1) <br>
[4] [Lung-RADS assessment categories. 2014. The American College of Radiology.](https://www.acr.org/-/media/ACR/Files/RADS/Lung-RADS/LungRADS_AssessmentCategories.pdf?la=en) <br>
[5] [Cohens' kappa - Wikipedia](https://en.wikipedia.org/wiki/Cohen%27s_kappa) <br>
[6] Ciompi, F. et al. Towards automatic pulmonary nodule management in lung cancer screening with deep learning. Nature Reviews Cancer. <br>
[7] [Onno Mets, Robin Smithuis. 2017 Guideline for Pulmonary nodules. 2017. The Academical Medical Centre.](http://www.radiologyassistant.nl/en/p5905aff4788ef/fleischner-2017-guideline-for-pulmonary-nodules.html) <br>
[8] [강은영. 저선량 흉부 컴퓨터단층촬영을 이용한 폐암 선별검사: 영상의학 측면의 최신지견. 대한의사협회지.2015.](https://synapse.koreamed.org/pdf/10.5124/jkma.2015.58.6.523) <br>

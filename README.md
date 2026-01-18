# Do-Novo: Observed Spectrum Driven Do-Sampling of Theoretical Fragments for De Novo Peptide Sequencing in DIA
![img](./figures/model_overview.png)

## Abstract
...


## How to run the experiments
### Prerequisites
This project is tested with following environments:
- Python: 3.11.13
- CUDA: 12.6
- torch: 2.8.0+cu126
- Pytorch-lightning: 2.5.5
- pyteomics: 4.7.5
- depthcharge-ms: 0.4.8
---
### Install environment (Linux)
```
conda env create --file environment.yml
conda activate do_novo_env
```

---
### Download datasets
If you want to annotated dataset when training our model,
```
pip install gdown
gdown https://drive.google.com/uc?id=1LElJGJ9q9y1Q_iyfvtvre4V10mGDO1Y0 # datasets(oc, uti, plasma)
tar -zxvf data.tar.gz
```
---
### Training models
The easiest way to train a model is specify a config file (eg `configs/oc_train_ump.yaml`) with data, model, and training hyperparameters
```
python main.py --config ./configs/oc_train_ump.yaml
```

---
## Acknowledgement
This code includes modifications based on the code of Cascadia. We are grateful to the authors for providing their code/models as open-source software.

## Contributors

<table>
	<tr>
		<th>Name</th>		
		<th>Affiliation</th>
		<th>Email</th>
	</tr>
	<tr>
		<td>Seungheun Baek&dagger;</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>sheunbaek@korea.ac.kr</td>
	</tr>
	<tr>
		<td>Yan Ting Chok</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>yanting1412@korea.ac.kr</td>
	</tr>
	<tr>
		<td>Eunha Lee</td>
        <td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>eunhalee@korea.ac.kr</td>
	</tr>
	<tr>
		<td>Jaewoo Kang*</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>kangj@korea.ac.kr</td>
	</tr>
</table>

- &dagger;: *First Author*
- &ast;: *Corresponding Author*
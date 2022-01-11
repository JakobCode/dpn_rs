# An Advanced Dirichlet Prior Network for Out-of-Distribution Detection in Remote Sensing

## Out-of-Distribution detection for unseen classes, changing sensors, and changing locations.

The proposed method uses in-distribution and out-of-distribution training data in order to learn a gap between (familiar) in-distribution data and (partially unknown) out-of-distribution data. 



**To run the code** please download the datasets and put them in datasets folder.<br/>
Then cd to Experiments directory and run the code for the UCM dataset by using following commands:  <br/>
python experiment_sensor_shift.py --data ucm --approach dpn_rs --seed 42 <br/>
python experiment_class_shift.py --data ucm --approach dpn_rs --seed 42 <br/>
data, approach and seed can be adapted. 

### Citation
If you find this code useful, please consider citing:
```[bibtex]
@article{gawlikowski2022Andvanced,
  title={An Advanced Dirichlet Prior Network for Out-of-distribution Detection in Remote Sensing},
  author={Gawlikowski, Jakob and Saha, Sudipan and Kruspe, Anna and Zhu, Xiao Xiang},
  journal={IEEE Transactions in Geoscience and Remote Sensing}
  year={2022},
  publisher={IEEE}
}
```

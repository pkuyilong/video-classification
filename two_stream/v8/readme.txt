readme.txt

:Author: mayilong
:Email: mayilong@img
:Date: 2019-03-29 14:54


preprocess:
        select 16 frames seperately from flowx and flowy
        volume of data is smaller
        multi_scale
        resize        (256, 336)
        center crop   (224, 224) 

model: 
        res152
        dropout=0.3
        crossEntroy reducation=sum

# Cuboid_Abstraction
code of 'Learning Cuboid Abstraction of 3D Shapes via Iterative Error Feedback'

## 1) environment setting
Install a working implementation of pytorch and scipy.
## 2) data preprocessing
ShapeNet data is used in this project.Download ShapeNet v1(or v2) from www.shapenet.org/. And follow the data preprocessing step from https://github.com/shubhtuls/volumetricPrimitives. Put the processed data at **../data/xxxxxx/** as train set and  **../data/xxxxxx_t/** as test set.(xxxxxx should be the class index from shapeNet).Also put shapeNet data at **../shapenet/xxxxxx/**.
## 3) training
edit netName and classIndex in ./config.py.Then run **python train.py**.
## 4) testing
edit **test_net_name** in ./test.py.run **python test.py**. The result will show in ./cache/mat_cache/**test_net_name**/.

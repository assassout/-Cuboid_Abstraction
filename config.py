class config():
  netName = 'chair_init' # or netName = 'chair_init_test'
  classIndex ='03001627'
  dataDir = "../data/"+classIndex+'/'
  if_init = True
  loopInit = 20000
  ifLoadNet = False
  # 03001627 chair 02691156 airplane  02924116 bus 02828884 bench  04256520 sofa 04530566 ship 03636649 lamp
  #netName = 'bus_con_nosym'
  batchSize = 32
  gridBound = 0.5 
  celldan = 16
  moveBound = gridBound
  shapeBound = gridBound
  stepNum = 3
  gridsize = 32
  primNum = (gridsize // celldan) ** 3
  learningRateDecoder = 0.005
  LROrg = 0.001
  learningRatePretrain = 0.0001
  shapeLrDecay = 0.01
  transLrDecay = 0.1
  nSamplePoints = 10000
  nSamplesChamfer = 90
  nSamplesconfi = 300
  Nepochs = 1000
  datasize = 0 #change when data is load
  loopTime = 0+1
  numTrainIter = 20000
  cellGTSize = 4
  shapefeathers = 3 #3 for Cuboid ,2 for cylinder,1 for sphere


  loadNetName = netName
  loadNetIter = loopInit



  covWt = 1.0
  confiWt = 1.0
  consisWt = 1.0
  symWt = 1.0
  ifShapeOrg = False
  device = '1'

  dataDir_t = "../data/02691156_t/"
  testDir ="./test/"

  prim_intDir = "./cache/"
  netDir = "./cache/net_cache/"
  meshDir = "./cache/mesh_cache/"
  matDir = "./cache/mat_cache/"
  beaDir = "./cache/bea_cache/"
  meshSaveNum = 2 # how many mesh save in a patch
  meshSaveIter = datasize // 2
  netSaveIter = datasize // 2


  ifUseConfiLoss = True

  pointset=[[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]
  faceset=[[0,1,2],[1,3,2],[4,5,6],[5,7,6],
           [0,2,4],[4,2,6],[1,3,5],[5,3,7],
           [0,1,4],[4,1,5],[2,3,6],[6,3,7]]


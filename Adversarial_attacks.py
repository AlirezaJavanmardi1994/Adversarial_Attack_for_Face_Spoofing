!pip install foolbox
import foolbox as fb
import tensorflow as tf
bounds = (0,255)
fmodel = fb.TensorFlowModel(final_model,bounds = bounds)
labels = np.ones((len(images),),dtype=int)
labels = labels.astype('int32')
labels = tf.convert_to_tensor(labels)
import tensorflow as tf
images = tf.convert_to_tensor(images)
labels = tf.convert_to_tensor(labels)
########## Model ACC
fb.utils.accuracy(fmodel, images[:100],labels[:100])
########## First Attack FGSM
attack = fb.attacks.LinfFastGradientAttack()
########## First Attack PGD
attack = fb.attacks.LinfProjectedGradientDescentAttack()
########## First Attack DeepFool
attack = fb.attacks.LinfDeepFoolAttack()
########### Determining the effect of attack
################ Show Attack % ###############
qq = []
for i in range(150):
  raw, clipped, is_adv = attack(fmodel, images[i*10:(i+1)*10], labels[i*10:(i+1)*10], epsilons=5)
  for j in range(10):
    #if (is_adv[j] == True):
    qq.append(clipped[j])

qq = np.asarray(qq)
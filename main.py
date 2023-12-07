import tensorflow as tf
import matplotlib.pyplot as plt

# importando um dataset de numeros
mnist = tf.keras.datasets.mnist

# criando uma matriz com treinamentos e testes (treinamento = para treinar) (testes = ver se deu certo)
(train_images, train_labels) , (test_images, test_labels) = mnist.load_data()

# mostrando o dataset criado
print("train_images shape: ", train_images.shape)
print("train_labels shape: ", train_labels.shape)
print("test_images shape: ", test_images.shape)
print("test_labels shape: ", test_labels.shape)


# plotando o dataset
fig = plt.figure(figsize=(10,10))

nrows=3
ncols=3
for i in range(9):
  fig.add_subplot(nrows, ncols, i+1)
  plt.imshow(train_images[i])
  plt.title("Digit: {}".format(train_labels[i]))
  plt.axis(False)
plt.show()

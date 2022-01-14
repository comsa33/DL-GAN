# GAN
Generative Adversarial Network
## REFERENCE

- [https://ysbsb.github.io/gan/2020/06/17/GAN-newbie-guide.html](https://ysbsb.github.io/gan/2020/06/17/GAN-newbie-guide.html)

# Generative Adversarial Network : 대립생성신경망

---

> 기존의 GAN에 CNN의 네트워크를 도입한 DCGAN (Deep Convolutonal Generative Adversarial Networks)를 제안함. Supervised learning에서 CNN이 큰 역할을 하고 있는데, unsupervised learning에서는 CNN이 주목을 덜 받고 있었음. 이렇게 CNN에서 성공적인 점을 GAN에도 적용하여 기존 GAN보다 훨씬 좋은 성능을 내게 되었음. 그 전까지 GAN만 사용하면 성능이 좋지 않았으나, DCGAN 이후로부터 GAN의 발전이 많이 되었음.
> 
> ![Screen Shot 2022-01-14 at 14 15 13](https://user-images.githubusercontent.com/61719257/149455594-7945c343-0652-421e-8e36-8eb57e97028f.png)


## 1. GAN main idea

> Ian Goodfellow가 최초로 GAN (Generative Adversarial Nets)를 제안한 논문. 새로운 이미지를 생성하는 생성자 Generator와 샘플 데이터와 생성자가 생성한 이미지를 구분하는 구별자 Discriminator 두 개의 네트워크 구조를 제안함. 생성자는 구별자를 속이면서 이미지를 잘 생성하려고 하며, 구별자는 주어진 이미지자 진짜인지 가까인지 판별함.

**Generator : 생성자**
> 
> 
> > random noise 값으로부터 학습하는 이미지와 유사한 이미지를 생성하는 역할
> > 
> 
> **Discriminator : 판별자**
> 
> > Generator 로부터 생성된 이미지와 학습에 사용되는 진짜 이미지 중 무엇이 진짜인지 가짜인지 판별하는 역할
> > 

<aside>
📌 생성자가 세상에 있을 법한 가짜 이미지를 만들어 내면 판별자가 진짜 이미지와 비교하여 이를 판별하게 된다.
이 과정에서 생성자는 판별자를 속이기 위해 계속 향상된 이미지를 생성하게 되면서 훈련이 진행된다.

</aside>

## 2. Generator

```python
# generator
def make_generator_model():
    """
    생성자 함수
    """
    generator_input = Input(shape=(100,), name='generator_input')
    x = generator_input
    x = Dense(4*4*256, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Reshape((4,4,256))(x)
    
    x = UpSampling2D()(x)
    x = Conv2D(128, (3,3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = UpSampling2D()(x)
    x = Conv2D(64, (3,3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same', use_bias=False)(x)
    x = Conv2DTranspose(1, (3,3), strides=1, padding='same', use_bias=False, activation='sigmoid')(x)
    generator_output = x
    
    return tf.keras.models.Model(generator_input, generator_output)
```

upsampling 을 통해 랜덤한 noise로부터 이미지를 생성해 낸다. 

## 3. Discriminator

```python
# discriminator
def make_discriminator_model():
    """
    판별자 함수
    """
    discriminator_input = Input(shape=(32,32,1), name='discriminator_input')
    x = discriminator_input
    x = Conv2D(32, (4,4), strides=2, padding='same')(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(64, (4,4), strides=2, padding='same')(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(1)(x) # activation=sigmoid
    discriminator_output = x
    
    return tf.keras.models.Model(discriminator_input, discriminator_output)
```

단순한 CNN 구조이다. 입력받은 이미지가 진짜인지 생성된 가짜 이미지인지 판별하는 역할을 하게 된다.

최종 output 은 이진분류가 되어 0~1 사이 값을 갖게 된다.

## 4. Loss functions

```python
# loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    """판별자 손실함수"""
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    """생성자 손실함수"""
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
```

생성자와 판별자 두개의 모델이기 때문에 각각 손실함수가 존재한다.

### 판별자의 손실함수

> 진짜 이미지가 판별자를 거쳐 나온 output 은 1 이 되도록 손실함수를 구성하고
> 
> 
> 생성된 이미지가 판별자를 거쳐 나온 output 은 0 이 되도록 손실함수를 구성한다.
> 각각의 손실함수를 거친 값을 더해준다.
> 

### 생성자의 손실함수

> 생성자는 진짜처럼 보이도록 훈련이 되야하기 때문에 판별자를 거쳐나온 생성된 이미지의 output 이 1 이 되도록 손실함수를 구성해준다.
> 

## 5. Train

```python
@tf.function
def train_step(images):
    
    noise = tf.random.uniform([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    GenWeights = generator.trainable_variables
    DiscWeights = discriminator.trainable_variables
    
    gradients_of_generator = gen_tape.gradient(gen_loss, GenWeights)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, DiscWeights)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, GenWeights))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, DiscWeights))
```

```python
# display generated images
def generate_and_save_images(model, epoch, test_input):
    
    predictions = model(test_input, training=False)
    
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(predictions[i, :, :, 0]*255, cmap='gray')
        plt.axis('off')
    plt.savefig('a_images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
```

```python
import time
from IPython import display 

# train GAN
def train(dataset, epochs):
    
    for epoch in range(epochs):
        start = time.time()
        
        for image_batch in dataset:
            train_step(image_batch)
        
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch+1, seed)
        
        if (epoch+1)%20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print(f'Time for epoch {epoch+1} is {time.time()-start} sec')
    
    display.clear_ouput(wait=True)
    generate_and_save_images(generator, epochs, seed)
```

## CycleGAN
![cycleGAN_structure](https://user-images.githubusercontent.com/61719257/148213058-d3e1ef9c-645e-4da2-bd22-86343fe1f89c.gif)

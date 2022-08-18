import streamlit as st
import pickle
import numpy as np
from PIL import Image
import urllib.request
# from utils import *
import time

# Loading the training model
pickle_in = open('../Classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@st.cache()

# Function to generate labels
def gen_labels():
  train = '../Data/Train'
  train_generator = ImageDataGenerator(rescale = 1/255)

  train_generator = train_generator.flow_from_directory(train,
                                                      target_size = (300,300),
                                                      batch_size = 32,
                                                      class_mode = 'sparse')

  labels = (train_generator.class_indices)

  return labels


# Function for prediction
def prediction(img):
  prediction = classifier.predict(img[np.newaxis, ...])
  # #print("Predicted shape",p.shape)
  # print("Probability:",np.max(prediction[0], axis=-1))
  predicted_class = gen_labels()[np.argmax(prediction[0], axis=-1)]
  return predicted_class


# The meain function
def main():
  html_temp = '''
  <div>
  <h2></h2>
  <center><h3>Please upload Waste Image to find its Category</h3></center>
  </div>
  '''
  st.markdown(html_temp, unsafe_allow_html=True)
  
  # Html body
  st.set_option('deprecation.showfileUploaderEncoding', False)
  opt = st.selectbox("How do you want to upload the image for classification?\n", ('Please Select', 'Upload image via link', 'Upload image from device'))
  if opt == 'Upload image from device':
      file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
      st.set_option('deprecation.showfileUploaderEncoding', False)
      if file is not None:
        image = Image.open(file)
  elif opt == 'Upload image via link':
    img = st.text_input('Enter the Image Address')
    image = Image.open(urllib.request.urlopen(img))
    
    if st.button('Submit'):
      show = st.error("Please Enter a valid Image Address!")
      time.sleep(4)
      show.empty()
  
  # if image is not None:
  st.image(image,caption = 'Uploaded Image')
  if st.button('Predict'):
    img = np.array(image.resize((300, 300)))
    img = np.array(img, dtype='uint8')
    img = np.array(img)/255.0
    st.success('Hey! The uploaded image has been classified as " {} waste " '.format(prediction(img)))
    print('Nice Prediction, Thank you')


if __name__ == '__main__':
      main()

#         # model = model_arc()
#         pickle_in = open('classifier.pkl', 'rb')
#         classifier = pickle.load(pickle_in)
#         # model.load_weights("../weights/model.h5")
        


#         prediction = classifier.predict(img[np.newaxis, ...])
#         st.success('Hey! The uploaded image has been classified as " {} waste " '.format(labels[np.argmax(prediction[0], axis=-1)]))
#         print('Nice Prediction')
# except:
#   st.info('Something is wrong')


# @st.cache()



# html_temp = '''
#     <div style =  padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
#     <center><h1>Garbage Segregation</h1></center>
    
#     </div>
#     '''

# st.markdown(html_temp, unsafe_allow_html=True)



# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.markdown(html_temp, unsafe_allow_html=True)
# opt = st.selectbox("How do you want to upload the image for classification?\n", ('Please Select', 'Upload image via link', 'Upload image from device'))
# if opt == 'Upload image from device':
#     file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
#     st.set_option('deprecation.showfileUploaderEncoding', False)
#     if file is not None:
#         image = Image.open(file)





# try:
#   if image is not None:
#     st.image(image,caption = 'Uploaded Image')
#     if st.button('Predict'):
#         img = preprocess(image)

#         # model = model_arc()
#         pickle_in = open('classifier.pkl', 'rb')
#         classifier = pickle.load(pickle_in)
#         # model.load_weights("../weights/model.h5")
        
 

#         prediction = classifier.predict(img[np.newaxis, ...])
#         st.success('Hey! The uploaded image has been classified as " {} waste " '.format(labels[np.argmax(prediction[0], axis=-1)]))
#         print('Nice Prediction')
# except:
#   st.info('Something is wrong')

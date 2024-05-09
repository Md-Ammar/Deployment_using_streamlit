import streamlit as st
from model import *

st.title('Iris classfication model!')

st.text("This is a simple model to predict the species of iris flowers.")
st.text("The dataset looks like this:")
st.write(df.head())

test_size = st.slider("Test size", 0.1, 1.0, 0.2)

st.button("Run model", on_click=run_model())

# running = st.status("Model trained successfully!", state="running")
# running.state="complete"

# st.text("Model score: "+str(score))

st.text("Lets make a prediction!")

sepal_length = st.slider('Sepal Length', 4.3, 7.9, 5.4)
sepal_width = st.slider('Sepal Width', 2.0, 4.4, 3.4)
petal_length = st.slider('Petal Length', 1.0, 6.9, 1.3)
petal_width = st.slider('Petal Width', 0.1, 2.5, 0.2)

prediction = None
def predict(sepal_length, sepal_width, petal_length, petal_width):
    global prediction
    prediction = Model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

st.button("Predict", on_click=predict(sepal_length, sepal_width, petal_length, petal_width))

# Current flower information
st.table({"Sepal Length": sepal_length, "Sepal Width": sepal_width, "Petal Length": petal_length, "Petal Width": petal_width})
st.text("The model predicts: " + str(prediction))
st.text("The predicted species is: "+data.target_names[prediction])
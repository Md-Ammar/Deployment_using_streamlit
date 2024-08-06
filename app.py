import streamlit as st
from model import *

#heading
st.title('Iris classfication model!')

st.text("This is a simple model to predict the species of iris flowers.")
st.text("The dataset looks like this:")
st.write(df.head())

st.markdown("Use inbuilt dataset or upload your own!")
data = st.file_uploader("Upload a file", type=["csv"])

#setting test size
c1, c2 = st.columns(2)
c1.text("Set the test size for the model")
test_size = c2.slider("Test size", 0.1, 1.0, 0.2)

score = None

if st.button("Run model"):
    run_model(test_size)
    st.success("Model trained successfully!")
    score = get_score()
    st.metric("Model score: ", value=score)

#predicting
def predict(sepal_length, sepal_width, petal_length, petal_width):
    global prediction
    prediction = Model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    
st.text("Lets make a prediction!")

c1, c2, c3, c4 = st.columns(4)

sepal_length = c1.slider('Sepal Length', 4.3, 7.9, 5.4)
sepal_width = c2.slider('Sepal Width', 2.0, 4.4, 3.4)
petal_length = c3.slider('Petal Length', 1.0, 6.9, 1.3)
petal_width = c4.slider('Petal Width', 0.1, 2.5, 0.2)

st.table({"Sepal Length": sepal_length, "Sepal Width": sepal_width, "Petal Length": petal_length, "Petal Width": petal_width})

if st.button("Predict", on_click=predict(sepal_length, sepal_width, petal_length, petal_width)):
    predict(sepal_length, sepal_width, petal_length, petal_width)
    st.text("The model predicts: " + str(prediction))
    st.text("The predicted species is: "+data.target_names[prediction])
import streamlit as st
import pandas as pd
import joblib

#cargamos el modelo
model = joblib.load('model/lgb_model.pkl')

#configuramos la pagina

st.set_page_config(page_title='Datapp ML deployment',layout='centered')

#css 
st.markdown(
    """
    <style>
    .stApp{
        background-color: #34495e;
        font-family: 'Arial', sans-serif;
    }
    <style>
    """,
    unsafe_allow_html=True
)

#Titulo
st.title('Predicción de Éxito de tu Aplicación')

#Descripcion
st.write('''
Introduce las características de tu aplicación deseada para saber que tan exitosa será en la play store
''')

# opciones para las categorias
categories = {
     'Educación':'Education',
     'Video Juegos':'Game',
     'Negocios':'Business',
     'Estilo de vida':'Lifestyle',
     'Finanzas':'Finance',
     'Salud':'Health',
     'Productividad': 'Productivity',
     'Social': 'Social',
     'Entretenimiento': 'Entertainment',
     'Viajes': 'Travel'
}
#opciones en espanol

categories_esp = list(categories.keys())

# Obtenemos la informacion del usuario
def get_user_input():
    #definimos los nombres de nuestras caracteristicas
    user_data = {}
    #obtenemos la informacion de nuestra variable categorica
    selected_category = st.selectbox('Selecciona la categcategoría oria deseada',categories_esp)
    user_data['Category'] = categories[selected_category]
    # Obtenemos la informacion
    user_data['Free'] = [1 if st.radio('¿Es Gratis?', ['Sí', 'No']) == 'Sí' else 0]

    if user_data['Free'] == 0:
        user_data['Price'] = 0
    else:
        user_data['Price'] = [st.number_input('Precio',min_value=0.0,value=0.0)]
    user_data['Size'] = [st.number_input('Tamaño de la app (en MB)',min_value=0.0,value=0.0)]
    user_data['Minimum Android'] = [st.number_input('Verion Minima de Android',min_value=0.0,value=0.0)]
    user_data['Ad Supported'] = [1 if st.radio('¿Admitir anuncios?', ['Sí', 'No']) == 'Sí' else 0]
    
    return user_data

def main():
    targets = ['Rating','Installs']
    #obtenemos la informacion del usuarioEducation                  
    data = get_user_input()
    # guardamos la informacion en un data set
    df = pd.DataFrame(data)
    df['Category'] = df['Category'].astype('category')
    #Mostrar los datos ingresados por el usuario
    st.subheader('Datos Ingresados')
    #combertimos los datos a espanol
    new_df = df.copy()
    reverse_categories = {v: k for k, v in categories.items()}
    new_df['Category'] = new_df['Category'].map(reverse_categories)
    new_df[['Free','Price','Ad Supported']] = new_df[['Free','Price','Ad Supported']].applymap(lambda x: 'Si' if x == 1 else 'No')
    st.write(new_df)


    #hacemos nuestras predicciones
    if st.button('Predecir'):
        prediction = model.predict(df)
        # imprimimos los resultados
        st.subheader('Resultados')
        #prediciones de Rating en barra de aceptacion
        predicted_rating = prediction[0][0]
        acceptance_percentage = (predicted_rating/5 *100) # calculamos el porcentaje de aceptacion
        #mostrar la barra de aceptacion
        st.write(f'Rating Predicho: {predicted_rating:.2f} (Aceptación: {acceptance_percentage:.2f}%)')
        st.progress(int(acceptance_percentage))
        #mostrar predicion de instalaciones
        predicted_installs = prediction[0][1]  # Suponiendo que el segundo valor es la cantidad de instalaciones
        st.write(f'Instalaciones Predichas: {predicted_installs:,.0f}')

if __name__ == '__main__':
    main()

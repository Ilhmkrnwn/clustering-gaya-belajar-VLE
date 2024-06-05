import streamlit as st
import pandas as pd
import pickle
from streamlit_option_menu import option_menu
from PIL import Image
import scipy as sp
import numpy as np

st.set_page_config(page_title="Welcome at Kurniawan's Skripsi!", layout="wide")


def dbscan_predicts(model, X):

    nr_samples = X.shape[0]

    y_new = np.ones(shape=nr_samples, dtype=int) * -1

    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[
                model.core_sample_indices_[shortest_dist_idx]]

    return y_new


with st.sidebar:
    choose = option_menu("Kurniawan's Skripsi", ["Dokumentasi", "Clustering Demo"],
                         icons=['journal-richtext', ],
                         menu_icon="arrow-down-right-square-fill", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#262730"},
        "icon": {"color": "white", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#808080"},
        "nav-link-selected": {"background-color": "#0e1117"},
    }
    )

# HALAMAN DOKUMENTASI
if choose == 'Dokumentasi':
    st.markdown("""
            # Tugas Akhir Data Science 
            **👈 Pilih menu yang telah tersedia pada navbar disamping**
            
            Platform ini merupakan hasil penelitian yang telah dikembangkan menggunakan konsep _Unsupervised Learning_ pada bidang _Educational Data Mining_.            
            
            **Disusun dan dikembangkan oleh:**
            - Nama: [Ilham Kurniawan]()
            - Jurusan/Fakultas: Informatika/Teknik 
            - Pertanyaan lebih lanjut: [yoiilham@gmail.com](mailto:yoiilham@gmail.com)
            
            **Hasil atau keluaran dari penelitian ini berupa:**

            1. Data preprocessing menggunakan dataset yang berasal dari https://analyse.kmi.open.ac.uk/open_dataset
            2. Analisa Cluster menggunakan Bouldin Score dan Silhouette Analysis
            3. Clustering Model menggunakan algoritma DBSCAN dan OPTICS Clustering
            4. Model Deployment menggunakan Streamlit Framework
        """)

elif choose == 'Clustering Demo':
    st.write("# Clustering Demo ")

    with st.form(key='values_in_form'):
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        with col1:
            code_module = st.selectbox(
                'Code Module', ('AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'))
        with col2:
            gender = st.selectbox('Gender', ('M', 'F'))
        with col3:
            disability = st.selectbox('Disability', ('N', 'Y'))
        with col4:
            final_result = st.selectbox('Final Result', ('Pass', 'Fail'))
        with col5:
            score = st.slider('Score', 1, 100, 1)
        col6, col7, col8, col9, col10 = st.columns([1, 1, 1, 1, 1])
        with col6:
            assessment_type = st.selectbox('Assessment Type', ('TMA', 'CMA'))
        with col7:
            activity_type = st.selectbox(
                'Activity Type', ('resource', 'oucontent', 'url', 'subpage'))
        with col8:
            total_activities = st.slider('Total Activities', 9, 36, 9)
        with col9:
            sum_click = st.slider('Sum Click', 1, 100, 1)
        submitted = st.form_submit_button("Predict Learning Style")

    data = [{
        'code_module': code_module,
        'gender': gender,
        'disability': disability,
        'final_result': final_result,
        'score': score,
        'assessment_type': assessment_type,
        'activity_type': activity_type,
        'total_activities': total_activities,
        'sum_click': sum_click,
    }]

    dbscan_predict = pd.DataFrame(data, index=[0])

    scaler = pickle.load(open('./PKL/scaler.pkl', 'rb'))
    dbscan = pickle.load(open('./PKL/dbscan.pkl', 'rb'))

    categorical_cols = ['code_module', 'gender', 'disability',
                        'final_result', 'assessment_type', 'activity_type']

    if submitted:
        print(dbscan_predict)
        for cat in categorical_cols:
            encoder = pickle.load(open('encoder/'+cat+'_encoder.pkl', 'rb'))
            dbscan_predict[cat] = encoder.transform(dbscan_predict[cat])
        print(dbscan)
        dbscan_predict = scaler.transform(dbscan_predict)
        print(dbscan_predict)
        prediction = dbscan_predicts(dbscan, dbscan_predict)
        print(prediction)
        if prediction == 0:
            pred1 = 'visual'
        elif prediction == 1:
            pred1 = 'aural'
        elif prediction == 2:
            pred1 = 'read'
        elif prediction == 3:
            pred1 = 'kinaesthetic'
        else:
            pred1 = 'noise'

        st.success('Your Learning Style is {}'.format(pred1))

    st.write("---")

    st.write("### Deskripsi Klaster tiap Gaya Belajar:")
    dec1, dec2, dec3, dec4 = st.columns([1, 1, 1, 1])
    with dec1:
        st.markdown("""
            1. Visual:
                - Mean score = 63
                - Mean total_activities = 9
                - Mean sum_click = 12
                - Lebih menyukai assement_type TMA (Tutor Marked Assesment)
                - Mayoritas memiliki Gender Perempuan
                - Sebagian besar Bukan Penyandang Disabilitas
                - Cenderung lebih banyak mendapatkan Final Status berupa Pass
        """)
    with dec2:
        st.markdown("""
            2. Aural:
                - Mean score = 61
                - Mean total_activities = 21
                - Mean sum_click = 15
                - Lebih menyukai assement_type CMA (Computer Marked Assesment)
                - Mayoritas memiliki Gender Perempuan
                - Sebagian besar Bukan Penyandang Disabilitas
                - Cenderung lebih banyak mendapatkan Final Status berupa Pass
        """)
    with dec3:
        st.markdown("""
            3. Read/Write:
                - Mean score = 65
                - Mean total_activities = 11
                - Mean sum_click = 18
                - Lebih menyukai assement_type TMA (Tutor Marked Assesment)
                - Mayoritas memiliki Gender Laki-Laki
                - Sebagian besar Bukan Penyandang Disabilitas
                - Cenderung lebih banyak mendapatkan Final Status berupa Pass
        """)
    with dec4:
        st.markdown("""
            4. Kinaesthetic:
               - Mean score = 64
                - Mean total_activities = 34
                - Mean sum_click = 23
                - Lebih menyukai assement_type CMA (Computer Marked Assesment)
                - Mayoritas memiliki Gender Laki-Laki
                - Sebagian besar Bukan Penyandang Disabilitas
                - Cenderung lebih banyak mendapatkan Final Status berupa Pass
        """)

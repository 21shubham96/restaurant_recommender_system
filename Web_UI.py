import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

resto = pickle.load(open('resto_name.pkl','rb'))
df3 = pickle.load(open('df4.pkl','rb'))
data = pickle.load(open('data.pkl','rb'))


list6 = []
def get_restro(selected_cuisine):
    dict1  = dict(zip(df3['Name'], zip(df3['0'],df3['1'],df3['2'],df3['3'],df3['4'],df3['5'])))
    
    
#     #Appending Restaurants Names in list6 using dictionary 
    for x, y in dict1.items():
        for yy in y:
            if yy == selected_cuisine:
                list6.append(x)
    
#     #Saving First Restaurant name in variable Resto_name
    Resto_name = list6[0]
    return Resto_name
    
def recommendation(resto_name):
    
    new_data = data[data["resto_name"].str.contains('|'.join(list6))]
    #Model
    
#     #Convert a collection of raw documents to a matrix of TF-IDF features.
    tfidf = TfidfVectorizer(stop_words='english')
    
#     #Learn vocabulary and idf, return document-term matrix
    tfidf_matrix = tfidf.fit_transform(new_data['new_review'].values.astype('str'))
    
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    
    indices = pd.Series(df3.index, index=df3["Name"]).drop_duplicates()
    
    
# # #     # Get the index of the Restaurants that matches the Restaurants Name
    idx = indices[resto_name]

# # #     # Get the pairwsie similarity scores of all Restaurants with that Restaurants
    sim_scores = list(enumerate(cosine_sim[idx]))

#     # Sort the Restaurants based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

# # #     # Get the scores of the 100 most similar Restaurants
    sim_scores = sim_scores[0:100]

# # # #     # Get the restaurant indices
    resto_indices = [i[0] for i in sim_scores]
    
    res = new_data['resto_name'].iloc[resto_indices]
    res = res.drop_duplicates()
    res = res.reset_index()
    res.drop(['index'], axis=1, inplace=True)
    
    return res[0:10]
    
        
    



def main():
    
    st.title('Restaurants: As Per Your Taste Buds')
    
    selected_cuisine= st.selectbox(
    ' ',
    ['Chinese', 'Biryani', 'Asian', 'Seafood', 'Healthy Food',
       'Lebanese', 'American', 'Ice Cream', 'Street Food',
       'Mediterranean', 'Continental', 'North Indian', 'European',
       'Burger', 'Mexican', 'Mughlai', 'Fast Food', 'Bakery', 'Arabian',
       'Italian', 'Andhra', 'Cafe', 'Finger Food', 'South Indian',
       'Modern Indian', 'Desserts', 'Kebab', 'BBQ', 'Momos',
       'North Eastern', 'Hyderabadi', 'Thai','Italian'])
    
    
    
#     cuisine = st.text_input('Cuisine Type')
    
    cusines = ''
    
    if st.button('Search'):
        resto_name = get_restro(selected_cuisine)
        cusines = recommendation(resto_name)
        
        st.dataframe(cusines)
        
        
if __name__ == '__main__':
    main()



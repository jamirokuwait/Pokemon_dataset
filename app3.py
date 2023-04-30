import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression


def main():
    ###############################################################
    kanto = Image.open('kanto.png')
    johto = Image.open('johto.png')
    hoenn = Image.open('hoenn.png')
    sinnoh = Image.open('sinnoh.png')
    unova = Image.open('unova.png')
    kalos = Image.open('kalos.png')
###############################################################
    st.title('Researching on Pokemon through Generations')

    df = pd.read_csv('Pokemon.csv')
    df = df.fillna(value='miao')
    dfgroup = df.set_index('Name')
    dfheatmapstat = dfgroup.drop(
        columns=['Type 1', 'Type 2', 'Generation', 'Legendary', '#', 'Total'], axis=1)
    dfheatmaptype = dfgroup.drop(columns=[
                                 'Legendary', '#', 'Total', 'HP', 'Attack', 'Defense', 'Speed', 'Sp. Atk', 'Sp. Def'], axis=1)
    # st.dataframe(dfheatmaptype)
    # st.dataframe(dfheatmap)

    X = dfgroup.drop(columns=['Type 1', 'Type 2', 'Generation',
                     'Legendary', '#'], axis=1)  # 'Type 1','Type 2',
    y = df['Type 1']
    st.write('Taking an overwiev on the dataset we are analyzing.')
    st.dataframe(X)
    st.write('This set includes also MegaEvolutions,we will discuss about it later.')
    ###############################################################
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(dfheatmapstat.corr(), annot=True)
    st.subheader('HEATMAP(Stats correlation)')
    st.write('Operating correlation (.corr()) between our datasets stats value return us this beatiful heatmap')
    st.pyplot(fig)
    st.write(
        'As you can see,between each different stat there is little correlation/Negative correlations(i.e. if defense is higher then speed will be lower,in the same way faster pkmn tend to have less Hp,sp. attacker have more probability of having higher sp.def than physical def,...)')
###############################################################
    gen1 = dfgroup[dfgroup['Generation'] == 1]
    gen2 = dfgroup[dfgroup['Generation'] == 2]
    gen3 = dfgroup[dfgroup['Generation'] == 3]
    gen4 = dfgroup[dfgroup['Generation'] == 4]
    gen5 = dfgroup[dfgroup['Generation'] == 5]
    gen6 = dfgroup[dfgroup['Generation'] == 6]
###############################################################
    st.subheader('Visualizing correlations')
    st.write('In this plots we have grouped our data by type/gen on the y axis,while on the x we have the amount of pokemon introduced. WHY SO MANY WATER MON?!(NB 31/27 FOR 1ST AND 3RD GEN,112 TOTAL)')

    fig = plt.figure(figsize=(10, 4))
    sns.countplot(data=df, y='Type 1',
                  order=df['Type 1'].value_counts().index)  #
    plt.ylabel('Pkmn Types 1')
    plt.xlabel('Number of Pokemon types')
    plt.title('PKMN type 1 over total')
    st.pyplot(fig)
    st.write('As we can clearly see in this plot,more than 100 pokemons(through 6 gen) are water type,followed by Normal (98),Grass(70) and bug(69)')
    st.write('Following the principle that every mon is related to his/her specific enviroment we can try to make some assumption about the enviroment:')
    st.write('- Water is the dominant Enviroment in the Pkmn world(more than 1/8 of the total is a water type)')
    st.write('- 98/800 entries are normal types,i assume that normal type pkmn are related to humans/breeded for specific task, and human enviroment/cities is/are the 2nd most dominant env. ')
    st.write('- 3rd and 4th place goes to Grass and Bug type,we can say that the third most influent enviroment is the forest/mountain in the pkmn world')

    fig = plt.figure(figsize=(10, 4))
    sns.countplot(data=df, y='Type 2',
                  order=df['Type 2'].value_counts().index)  #
    plt.ylabel('Pkmn Types 2')
    plt.xlabel('Number of Pokemon types')
    plt.title('PKMN type 2 over total')
    st.pyplot(fig)
    st.write(
        'Here we have the plot representing the distribution of type 2 in our dataset')
    st.write('- the most present type 2 is the flying one (type 1 flying pokemon were not introduced until 6 gen with noibat and noivern,all the flying pokemons from previous gen were simply normal pkmns)')
    st.write(
        '- Ground,Poison and Psychic are the 2nd most present type 2,with 35/34/33 entries')
    st.write(
        '- (Ground ones can be related to the high presence of mountains and caves,maybe they acquired the ground type 2 for living near/in these places. Psych and poison strong presence among type 2 types are related to a speculation that implies a war fought between regions[pokemon became poisonous/mutated because of biological weapons] and experimentations operated on these little monsters[and being used as weapons,maybe?])')

    fig = plt.figure(figsize=(10, 4))
    sns.countplot(data=df, y='Generation',
                  )  # order=df['Generation'].value_counts().index
    plt.ylabel('Pkm introduced by Gen')
    plt.xlabel('Number of Pokemon introduced')
    plt.title('PKMN Introduced each Gen(Gen)')
    st.pyplot(fig)
###############################################################
    with st.expander('Pokemon type introduced by each gen'):
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ['Gen 1', 'Gen 2', 'Gen 3', 'Gen 4', 'Gen 5', 'Gen 6'])
        with tab1:
            fig = plt.figure(figsize=(10, 4))
            sns.countplot(data=gen1, y='Type 1',
                          order=gen1['Type 1'].value_counts().index)
            plt.title('Gen 1')
            st.pyplot(fig)
        with tab2:
            fig = plt.figure(figsize=(10, 4))
            sns.countplot(data=gen2, y='Type 1',
                          order=gen2['Type 1'].value_counts().index)
            plt.title('Gen 2')
            st.pyplot(fig)
        with tab3:
            fig = plt.figure(figsize=(10, 4))
            sns.countplot(data=gen3, y='Type 1',
                          order=gen3['Type 1'].value_counts().index)
            plt.title('Gen 3')
            st.pyplot(fig)
        with tab4:
            fig = plt.figure(figsize=(10, 4))
            sns.countplot(data=gen4, y='Type 1',
                          order=gen4['Type 1'].value_counts().index)
            plt.title('Gen 4')
            st.pyplot(fig)
        with tab5:
            fig = plt.figure(figsize=(10, 4))
            sns.countplot(data=gen5, y='Type 1',
                          order=gen5['Type 1'].value_counts().index)
            plt.title('Gen 5')
            st.pyplot(fig)
        with tab6:
            fig = plt.figure(figsize=(10, 4))
            sns.countplot(data=gen6, y='Type 1',
                          order=gen6['Type 1'].value_counts().index)
            plt.title('Gen 6')
            st.pyplot(fig)
###############################################################
    st.write('- In this plot we see that the pokemon distribution among the Gens is not equal,1/3/5 gen have more pokemons than the even gens(2/4/6)')
    st.write('- this can be related to a focus from developers on story/characters rather than collecting Pokemons(more monsters means more time devoted to capture them)')

    fig = plt.figure(figsize=(10, 15))
    sns.countplot(data=df, y='Type 1', hue='Generation',
                  order=df['Type 1'].value_counts().index)  #
    plt.ylabel('Pkm Type 1 introduced')
    plt.xlabel('Number of Pokemon introduced by Gen')
    plt.title('PKMN Introduced each Gen(Type 1)')
    st.pyplot(fig)
    st.write('- With this plot we can desume some other important infos.')
    st.write('- until 4th gen water monsters were the most present type in each gen(1-2-3),until Normal types took their place(also in 5th gen we have more Normal than other types)')
    st.write(
        '- another strage fact is that in the 6th gen we have more ghost monsters than other types')
    st.write(
        '- (Fun fact: Ground/Rock ones are related with rocky/caves/mountains enviroment of the world[5th,1st and 3rd gen have the highest amount of ground/rock pkmns,related to mountains and caves in these 2 regions,also dark types have their highest presence in gen 3/5,i think its also related to dark places like caves])')

    fig = plt.figure(figsize=(10, 15))
    sns.countplot(data=df, y='Type 2', hue='Generation',
                  order=df['Type 2'].value_counts().index)  #
    plt.ylabel('Pkm Type 2 introduced')
    plt.xlabel('Number of Pokemon introduced by Gen')
    plt.title('PKMN Introduced each Gen(Type 2)')
    st.pyplot(fig)
###############################################################
    with st.sidebar:
        gen_ = st.selectbox('Select Gen:', options=df['Generation'].unique())
        type_ = st.selectbox('Select 1st Type:', options=df['Type 1'].unique())
###############################################################
    type1df = df[df['Type 1'] == type_]
    totgen = df[df['Generation'] == gen_]

    gendf = type1df[type1df['Generation'] == gen_]

    val = gendf['Type 1'].count()  # tot tipo pokemon per gen
    total = df['Name'].count()  # tot pokemon per gen
    totalt = type1df['Name'].count()  # tot pokemon tipo
    totalg = totgen['Generation'].count()  # tot pokemon per gen
###############################################################
    st.subheader(
        'Evaluating pokemons by type and/or generation')
    st.write('Here you can take a quick look at the Map for selected gen:')
    with st.expander('Click to display the current gen region map'):
        if gen_ == 1:
            st.image(kanto, caption='Kanto region(1st Gen)')
        if gen_ == 2:
            st.image(johto, caption='Johto region(2nd Gen)')
        if gen_ == 3:
            st.image(hoenn, caption='Hoenn region(3rd Gen)')
        if gen_ == 4:
            st.image(sinnoh, caption='Sinnoh region(4th Gen)')
        if gen_ == 5:
            st.image(unova, caption='Unova region(5th Gen)')
        if gen_ == 6:
            st.image(kalos, caption='Kalos region(6th Gen)')
    st.write('(We can use it as an enviromental reference for addressing correlations between regional enviroment and gen composition)')
    st.write('You choose', type_, 'pokemons from the sidebar.')
    st.write('The number of', type_,
             ' types in the', gen_, 'Generation is ', val, '. (Out of', totalg, 'pkmns this Gen has introduced,including Megas.)')
    st.write('(We can presume that the specific amount of pokemon types in each gen can be correlated to the enviroment of the region.)')
    st.write('Pokemon of choosen type and generation are listed below.')
    with st.expander('Click to expand'):
        st.dataframe(gendf)

    st.write('Here we have the Pokemons listed only by the choosen type(type 1)')
    with st.expander('Click to expand'):
        st.write('For', type_, 'Pokemons we have a total of',
                 totalt, 'creatures among', total)
        st.dataframe(type1df)

        fig = plt.figure(figsize=(12, 6))
        sns.countplot(data=type1df, y='Type 1', hue='Generation')
        st.pyplot(fig)

    st.write('a selector for type 2, f l a v o u r .')

    with st.expander('Type 2 selector'):
        type_2 = st.selectbox('Select 2nd Type:',
                              options=df['Type 2'].unique())
        type2df = df[df['Type 2'] == type_2]
        total2 = type2df['Name'].count()
        st.write('Total Number of', type_2, 'pokemons is', total2)
        st.dataframe(type2df)
###############################################################
    pokemon_stats_by_generation = dfgroup.groupby('Generation').mean(
    )[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
    pokemon_stats_by_type = dfgroup.groupby('Type 1').mean(
    )[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
    pokemon_stats_by_hp = dfgroup.groupby('Type 1').mean(
    )[['HP']]
    pokemon_stats_by_atk = dfgroup.groupby('Type 1').mean(
    )[['Attack']]
    pokemon_stats_by_def = dfgroup.groupby('Type 1').mean(
    )[['Defense']]
    pokemon_stats_by_spatk = dfgroup.groupby('Type 1').mean(
    )[['Sp. Atk']]
    pokemon_stats_by_spdef = dfgroup.groupby('Type 1').mean(
    )[['Sp. Def']]
    pokemon_stats_by_sp = dfgroup.groupby('Type 1').mean(
    )[['Speed']]
###############################################################
    st.subheader('Analize Gens and Stats')
    # pokemon_stats_by_type = pokemon_stats_by_type.sort_values(by='HP')
    st.write(
        '- We have grouped our monsters by gen and then calc the mean value of all stats among gens...')
    st.dataframe(pokemon_stats_by_generation)
    st.write('...and then plotted below for visualize our data in a fashionable way.')

    fig = plt.figure(figsize=(10, 8))
    sns.lineplot(data=pokemon_stats_by_generation)
    plt.ylabel('Mean stats value for each gen')
    st.pyplot(fig)

    st.write('- As we can clearly see from above,from 1st to 2nd gen we have a 3 stats that decrease(attack,speed and sp.atk) and 3 that increase their mean value(def,sp.def and hp). We can say that 2nd gen pkmn are more tanky ')
    st.write('- From 2nd to 3rd we see that atk and sp.atk stats increase in a significant way(reach higher level than 1st gen),speed and def also slightly increase their mean value. Sp.def and Hp fall down,but not in a significant way this time(lower than 2nd gen but way higher than 1st). 3rd gen monster are surely good attackers.')
    st.write('- From 3rd to 4th gen all stats increase their mean values,with all stats reaching new peak values(speed mean increase but his under the speed cap setted in the 1st gen). We can say that 4th gen are surely stronger than previous gen but not faster.')
    st.write('- The 5th gen has realigned the stats from the previous gen: they have loweered all the stats. They have dropped significantly except atk and hp,they have slightly decreased. ')
    st.write('- The 6th gen resetted the tanky trend from the second gen,sort of. def,sp def and sp atk increase while atk(who dropped significantly),hp and speed decreased. ')

    with st.expander('click to see dataframes for every gen and detailed gen analysis'):
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ['Gen 1', 'Gen 2', 'Gen 3', 'Gen 4', 'Gen 5', 'Gen 6'])
        with tab1:
            st.dataframe(gen1)
            st.write('- Besides Mega evolutions and legendary pokemon,the powerful pokemon of the first gen is Dragonite,with stats total of 600,2nd place for Arcanine(555),3rd for Snorlax and Gyarados(540),for total stats. All the pokemon listed are physical attackers.')
            st.write('- Most hp award goes to..Chansey!but..with a total of 450 (that is not that bad) the sum of atk,def and sp atk is only 45. Snorlax is the 2nd in the list,with stats way better than chansey')
            st.write('- We got a hint from the previous plot: Atk in the 1st gen was the highest stats in value,followed by speed,sp atk,def,sp def and lastly hp.')
            st.write(
                '- Dragonite best attacker(134),Cloyster best defender(180),sp. attacker?Alakazam,obv(135). Mr. Mime and Tentacruel for a solid 120 in sp def and Electrode(a rolling ball) has the highest speed (140). ')
            st.write('- With 130 atk stats each we have 4 different pokemon: Flareon(fire,525 tot),Machamp(fighting,505 tot),Rhydon(ground,485 tot) and Kingler(water,475 tot). These are all evolved monsters')
            st.write('- In the best defender list,below Cloyster,we have 3 rock type pokemons(correlations between type and stat): Golem,Omastar(evolved ones,495 tot stats,130 and 125 def stat) and Onix (160 def,395 tot stats)')
            st.write('- For sp atk we have 2 of the most iconic mons of the series: Alakazam(135 with 500 tot stats) and Gengar (130 with 500 tot stats),the first is more faster and has a better sp.def,the ghost one has general slightly better stats (in a certain way is less glasscannon than his friend)')
            st.write(
                '- Below Electrode in the speed list we find Aerodactyl(fossil pokemon) and Jolteon,with good overall stats,above 500')
        with tab2:
            st.dataframe(gen2)
            st.write('- In the second gen they tried to set pkmn power almost similarly to the previous gen: game specific legendary (Lugia and Ho-oh,in the first gen there were Mew and Mewtwo) have their total set to 680,while sub-legendary pkmn(the three dogs,like the three birds from Kanto region) have 580.')
            st.write(
                '- Tyranitar is now the strongest of the actual gen,with an astonishing total of 600,like Dragonite.')
            st.write(
                '- As we can see the tot stats trend follow the previous gen: three starters fully evolved range from 534 to 525,slightly below top tier monsters of this gen(Blissey[Chansey evolution,540],Kingdra[540] and Crobat[535]). Also the second gen eeveelutions (eevees evolutions,ndr) have the same total (525).')
            st.write('- For the best attackers award we see Tyranitar in the first place(134,rock type),followed by Ursaring(130,normal type) and 2 bug type monsters: Scizor(130 with 500 tot stats) and Heracross(also 500 with 125 atk)')
            st.write('- Best 2nd gen defender and sp defender is Shuckle(bug type with 505 tot),230 in each defense stats! Followed by Steelix(steel evolution of onix),200 def with overall better stats')
            st.write('- Espeon is the best sp attacker of this gen,with 130 sp atk (and 110 speed stat that is on of the highest of johto monsters)')
            st.write('- Faster pokemon of this gen is..Crobat,the Zubat final form,130 speed stat,poison/flying type. Only Sneasel is faster than Espeon.')
        with tab3:
            st.dataframe(gen3)
            st.write('- Things are getting curious in Hoenn region: Legendary trend is settled as usual,this thime with more legendary pokemons(680/670 tot stats for franchise legendary,600 tot for "mid" legendary like latios and latias,580 for the regis)')
            st.write('- But here things become slightly different from the previous gens: we have a non-leg pokemon with a total of 670(Slaking,a normal type physical attacker,IF he attacks he does a lot of damage but he doesnt attack every turn due his ability) and 2 600 tot stats monsters(Metagross and Salamance,steel/psychic and dragon/flying,still used today in competitve tournaments,135 atk stat each,below Slaking that has 160)')
            st.write('- Starters follow the previous pattern,ranging from 535 to 530 now(slightly better stats),water types are now in higher position than the 2nd gen(in the first 20 non leg we have 6 water type instead of 3 of the 2nd)')
            st.write(
                '- As we previously stated,the 3rd gen has better overall stats than 2nd,and this can be related with adding more legendary to the rooster(10 instead of 6,usually they tend to have better general stats)')
            st.write('- Something strange happened: grass and a ghost physical attacker in top tier atk stat!Breloom(grass,130),Cacturne(grass,115) and Banette(ghost,115). ')
            st.write('- The defense trend goes higher and higher! This gen monsters have an overall defense better than the 2 previous gens: in this gen we have good def stat distributed among more pokemons,instead of having less but tankier pokemons.')
            st.write('- The best non leg sp attacker of this gen is Gardevoir(Psychic,125),surely the higher presence of legendary monsters have raised the mean value of this gen sp. atks')
            st.write('- In gen 3 sp. def is slightly lower,')
        with tab4:
            st.dataframe(gen4)
        with tab5:
            st.dataframe(gen5)
        with tab6:
            st.dataframe(gen6)

    st.dataframe(pokemon_stats_by_type)
    fig = plt.figure(figsize=(14, 6))
    sns.lineplot(data=pokemon_stats_by_type)
    plt.ylabel('Mean stats value for each type')
    st.pyplot(fig)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ['Hp', 'Attack', 'Defense', 'Sp.Atk', 'Sp.Def', 'Speed'])
    with tab1:
        fig = plt.figure(figsize=(14, 6))
        sns.lineplot(data=pokemon_stats_by_hp)
        plt.ylabel('Mean HP value for each type')
        st.pyplot(fig)
    with tab2:
        fig = plt.figure(figsize=(14, 6))
        sns.lineplot(data=pokemon_stats_by_atk)
        plt.ylabel('Mean ATK value for each type')
        st.pyplot(fig)
    with tab3:
        fig = plt.figure(figsize=(14, 6))
        sns.lineplot(data=pokemon_stats_by_def)
        plt.ylabel('Mean DEF value for each type')
        st.pyplot(fig)
    with tab4:
        fig = plt.figure(figsize=(14, 6))
        sns.lineplot(data=pokemon_stats_by_spatk)
        plt.ylabel('Mean SPATK value for each type')
        st.pyplot(fig)
    with tab5:
        fig = plt.figure(figsize=(14, 6))
        sns.lineplot(data=pokemon_stats_by_spdef)
        plt.ylabel('Mean SPDEF value for each type')
        st.pyplot(fig)
    with tab6:
        fig = plt.figure(figsize=(14, 6))
        sns.lineplot(data=pokemon_stats_by_sp)
        plt.ylabel('Mean SPEED value for each type')
        st.pyplot(fig)
    # https://www.kaggle.com/datasets/abcsds/pokemon/code?resource=download

    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                         test_size = 0.25,
    #                                                         random_state = 667
    #                                                         )
if __name__ == '__main__':
    main()

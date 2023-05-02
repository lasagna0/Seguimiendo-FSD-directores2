import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import plotly
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import geopandas as gpd
from shapely.geometry import Point, Polygon
import json
from esridump.dumper import EsriDumper
import nbformat 


baqBARRIO=gpd.read_file('baqBARRIO.shp')
baqlocalidad=gpd.read_file('baqlocalidad.shp')
ipm_baqlocalidad_mean = gpd.read_file('ipm_baqlocalidad_mean.shp')
ipm_baqbarrio = gpd.read_file('ipm_baqbarrio.shp')
ipm_baqbarrio.head()
ipm_baqbarrio.rename(columns={'index_righ':'BARRIO'}, inplace=True)
ipm_baqbarrio.drop(columns=['embarazo_a','reactivaci', 'LABEL'], inplace=True)
ipm_baqbarrio_mean=ipm_baqbarrio.groupby('BARRIO').agg({'ipm':'mean'})
ipm_baqbarrio_mean=gpd.GeoDataFrame(ipm_baqbarrio_mean, geometry=baqBARRIO.geometry)
ipm_baqbarrio_meanwithlocalidad=ipm_baqbarrio_mean.merge(baqBARRIO, left_on='BARRIO', right_on='BARRIO', how='inner')
ipm_baqbarrio_meanwithlocalidad=gpd.GeoDataFrame(ipm_baqbarrio_meanwithlocalidad, geometry=baqBARRIO.geometry)
ipm_baqbarrio_meanwithlocalidad=ipm_baqbarrio_meanwithlocalidad[ipm_baqbarrio_meanwithlocalidad.geometry.notnull()]
ipm_baqbarrio_meanwithlocalidad=gpd.GeoDataFrame(ipm_baqbarrio_meanwithlocalidad, geometry=ipm_baqbarrio_meanwithlocalidad.geometry)





hide_streamlit_style = """

            <style>
           
             footer {
	
	visibility: hidden;
	
	}
footer:after {
	content:'Creado por David Sanchez Polo - Probarranquilla 2022'; 
	visibility: visible;
	display: block;
	position: relative;
	#background-color: red;
	padding: 5px;
	top: 2px;
            </style>

          
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
 
#function to define the app_layout
def app_layout():
    st.title("Analisis de datos para toma de decisiones - Agenda Social")
    st.write("Fecha de presentacion: "+str(pd.to_datetime("today")))
    st.write("La razón de esta documentación es buscar consenso en la toma de decisiones para el proceder de acción social liderado por la gerencia de agenda Intersectorial de Probarranquilla y la mesa directiva de empresarios. El siguiente buscara la representación y un análisis sintetizado de los datos disponibles sobre las regiones afectadas")
    st.subheader("Analisis a nivel de pobreza multidimensional (IPM)")
    
    st.write("La distribución de IPM a nivel de manzana, permite conocer el dato a nivel de manzana. Para una identificación correcta de los barrios, se calcula la situación de estos mismos y su localidad identificada. ")
   
    def df_filter_map(message,df):
        df_filter=st.slider("IPM", min_value=df['ipm'].min(), max_value=df['ipm'].max(), value=float(df['ipm'].min()), step=0.1)
        df=df[df['ipm']>=df_filter]
        fig2=plotly.graph_objects.Figure()
        fig2=px.choropleth_mapbox(df, geojson=baqBARRIO.geometry, locations=df.index, color="ipm",
                                color_continuous_scale="Viridis",
                                range_color=(ipm_baqbarrio_meanwithlocalidad['ipm'].min(), ipm_baqbarrio_meanwithlocalidad['ipm'].max()),
                                mapbox_style="carto-positron",
                                ## barranquilla
                                zoom=11, center = {"lat": 10.988, "lon": -74.803},
                                opacity=0.8,
                                labels={'ipm':'IPM', 'LOCALIDAD':'Localidad'},
                                title='Indice de pobreza multidimensional por barrio',
                                hover_name='BARRIO',
                                hover_data={'ipm':True, 'LOCALIDAD':True}

                                )
        FIG2=fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(FIG2)

        # show table with data
        df.reset_index(inplace=True)
        dfTAB=df[['BARRIO','ipm', 'LOCALIDAD']]
        dfTAB.columns=['Barrio','IPM', 'Localidad']
    
     
        ## TABLE
        gb=GridOptionsBuilder.from_dataframe(dfTAB)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gridOptions=gb.build()
        grid_response=AgGrid(dfTAB, gridOptions=gridOptions,
                            height=400,
                            width='100%',
                            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                            fit_columns_on_grid_load=True,
                            allow_unsafe_jscode=True,
                            )
        dfTAB=grid_response['data']
        selected=grid_response['selected_rows']
        selected=pd.DataFrame(selected)
        
    df_filter_map('IPM',ipm_baqbarrio_meanwithlocalidad)

    st.subheader("Analisis a nivel de pobreza multidimensional (IPM) por localidad")

    fig3=plotly.graph_objects.Figure()
    fig3=px.choropleth_mapbox(ipm_baqlocalidad_mean, geojson=baqlocalidad.geometry, locations=ipm_baqlocalidad_mean.index, color='ipm',
                            color_continuous_scale="Viridis",

                            range_color=(ipm_baqlocalidad_mean['ipm'].min(), ipm_baqlocalidad_mean['ipm'].max()),
                            mapbox_style="carto-positron",
                            ## barranquilla
                            zoom=11, center = {"lat": 10.988, "lon": -74.803},
                            opacity=0.8,
                            labels={'ipm':'IPM'},
                            title='Indice de pobreza multidimensional por localidad',
                            hover_name='index_righ',
                                hover_data={'ipm':True}
                            )

    FIG3=fig3.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(FIG3)
    fig3=plotly.graph_objects.Figure()
    ## map of suroccidente barrios
    just_suroccidente=ipm_baqbarrio_meanwithlocalidad[ipm_baqbarrio_meanwithlocalidad['LOCALIDAD']=='SUROCCIDENTE']
   

    st.write('Cuando minimizamos el detalle, encontramos que el area metroplitana es aquella con el IPM mas alto, no obstante, la frecuencia de barrios que se obtiene por encima de 20 en IPM para suroccidente es mayor que en la localidad Metropolitana.')

    ipm_baqbarrio_meanwithlocalidad__suroccidente=ipm_baqbarrio_meanwithlocalidad[ipm_baqbarrio_meanwithlocalidad['LOCALIDAD']=='SUROCCIDENTE']
    ipm_baqbarrio_meanwithlocalidad__suroccidente=ipm_baqbarrio_meanwithlocalidad__suroccidente[ipm_baqbarrio_meanwithlocalidad__suroccidente['ipm']>=20]
    ipm_baqbarrio_meanwithlocalidad__suroccidente=ipm_baqbarrio_meanwithlocalidad__suroccidente.drop(columns=['geometry'])
    ipm_baqbarrio_meanwithlocalidad__metropolitana=ipm_baqbarrio_meanwithlocalidad[ipm_baqbarrio_meanwithlocalidad['LOCALIDAD']=='METROPOLITANA']
    ipm_baqbarrio_meanwithlocalidad__metropolitana=ipm_baqbarrio_meanwithlocalidad__metropolitana[ipm_baqbarrio_meanwithlocalidad__metropolitana['ipm']>=20]
    ipm_baqbarrio_meanwithlocalidad__metropolitana=ipm_baqbarrio_meanwithlocalidad__metropolitana.drop(columns=['geometry'])

    ipm_baqbarrio_meanwithlocalidad__suroccidente=pd.DataFrame(ipm_baqbarrio_meanwithlocalidad__suroccidente)
    ipm_baqbarrio_meanwithlocalidad__suroccidente=ipm_baqbarrio_meanwithlocalidad__suroccidente[['BARRIO','ipm']]
    ipm_baqbarrio_meanwithlocalidad__metropolitana=pd.DataFrame(ipm_baqbarrio_meanwithlocalidad__metropolitana)
    ipm_baqbarrio_meanwithlocalidad__metropolitana=ipm_baqbarrio_meanwithlocalidad__metropolitana[['BARRIO','ipm']]
    st.write('Barrios con IPM mayor a 20 en Suroccidente')
    gb=GridOptionsBuilder.from_dataframe(ipm_baqbarrio_meanwithlocalidad__suroccidente)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gridOptions=gb.build()
    grid_response=AgGrid(ipm_baqbarrio_meanwithlocalidad__suroccidente, gridOptions=gridOptions,
                        height=400,
                        width='100%',   
                        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,    
                        fit_columns_on_grid_load=True,
                        allow_unsafe_jscode=True,   
                        )   
    ipm_baqbarrio_meanwithlocalidad__suroccidente=grid_response['data']
    selected=grid_response['selected_rows']
    
    st.write('Barrios con IPM mayor a 20 en Metropolitana')
    gb=GridOptionsBuilder.from_dataframe(ipm_baqbarrio_meanwithlocalidad__metropolitana)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gridOptions=gb.build()
    grid_response=AgGrid(ipm_baqbarrio_meanwithlocalidad__metropolitana, gridOptions=gridOptions,
                        height=400,
                        width='100%',   
                        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                        fit_columns_on_grid_load=True,
                        allow_unsafe_jscode=True,
                        )
    ipm_baqbarrio_meanwithlocalidad__metropolitana=grid_response['data']
    selected=grid_response['selected_rows']
    st.header('Analisis de Barrios para suroccidente')
    st.write('Por la narrativa anterior, de un mayor impacto al adentrarse a comunidades del suroccidente, se procede a analizar el comportamiento de los barrios en esta localidad.')
    df_filter_map('IPM', just_suroccidente)
    st.write('Es por ello por lo que haciendo una organización de mayor a menor basado en el nivel de problemáticas por IPM se prioriza los barrios 7 de Agosto, Cuchillas del Villate y El Golfo y El Bosque. A pesar de su alto nivel de IPM, los barrios Pinar Del Rio y Villas de la Cordialidad son eliminados de esta priorización por su carácter de ser invasiones. ')
    colegios_suroccidente=gpd.read_file('colegios.shp')
    colegios_suroccidente['lat']=colegios_suroccidente['geometry'].apply(lambda x: x.y)
    colegios_suroccidente['lon']=colegios_suroccidente['geometry'].apply(lambda x: x.x)
    ## GRAPH BARRIOS POLYGONS AND COLEGIOS POINTS

    st.header('Analisis de equipamiento de barrios afectados')
    st.write('Para poder obtener un primer acercamiento con las comunidades presentes en aquellos barrios con mayor afectación, e identificar problemáticas arraigadas a estos. Mediante análisis espaciales determinamos cuales barrios tienen un colegio dentro de sus limitaciones, y en el caso que estos no lo tengan, se usara métodos de minimización de espacio, teniendo en cuenta la formulación de Haversine ("Virtues of the Haversine, 1984). Para determinar el colegio más cercano a cada barrio.')
    map2=px.choropleth_mapbox(just_suroccidente, geojson=just_suroccidente.geometry, locations=just_suroccidente.index, color='ipm',
                                color_continuous_scale="Viridis",
                                range_color=(ipm_baqbarrio_meanwithlocalidad['ipm'].min(), ipm_baqbarrio_meanwithlocalidad['ipm'].max()),
                                mapbox_style="carto-positron",
                                ## barranquilla
                                zoom=11, center = {"lat": 10.988, "lon": -74.803},
                                opacity=0.8,
                                labels={'ipm':'IPM', 'LOCALIDAD':'Localidad'},
                                title='Indice de pobreza multidimensional por barrio',
                                hover_name='BARRIO',
                                hover_data={'ipm':True, 'LOCALIDAD':True}
                                )

    map2.update_layout(margin={'r':0,'t':0,'l':0,'b':0})
    
    map2.add_trace(go.Scattermapbox(
        lat=colegios_suroccidente['lat'],
        lon=colegios_suroccidente['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(
        size=9
        ),
        text=colegios_suroccidente['NOMBRE_SED'],
        hoverinfo='text'

    ))
    st.plotly_chart(map2)
    
    conteodecolegios=gpd.sjoin(just_suroccidente, colegios_suroccidente, how='left', op='intersects')
    conteodecolegios2=conteodecolegios[['NOMBRE_SED', 'BARRIO_left', 'geometry', 'ipm']]

    conteodecolegios=conteodecolegios[['NOMBRE_SED', 'BARRIO_left', 'geometry', 'ipm']]
    grupodecolegios=conteodecolegios
    conteodecolegios=conteodecolegios.groupby('BARRIO_left').count()
    conteodecolegios['ipm']=conteodecolegios2.groupby('BARRIO_left').agg({'ipm':'mean'})
    conteodecolegios.columns=['conteo', 'geometry', 'ipm']
    
    conteodecolegios.fillna(0,inplace=True)
    conteodecolegios.sort_values(by=['conteo'], ascending=False, inplace=True)

## graph conteodecolegios with bar chart and ipm with points

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=conteodecolegios.index,
        y=conteodecolegios['conteo'],
        name='conteo',
        marker_color='indianred'
    ))

    fig.add_trace(go.Scatter(
        x=conteodecolegios.index,
        y=conteodecolegios['ipm'],
        name='ipm',
        mode='markers+lines',
        marker_color='lightsalmon'
    ))
    fig.update_layout(
        title='Conteo de colegios por barrio',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='conteo',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        )
    )
    ## x names set transversal
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)

    # Haversine Distance

    import math

    def    haversine(lat1, lon1, lat2, lon2):
        R = 6372800  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    



    def find_neareast_COLE(lat, lon):
        distances = colegios_suroccidente.apply(lambda row: haversine(lat, lon, row['lat'], row['lon']), axis=1)
        return colegios_suroccidente[distances == distances.min()]['NOMBRE_SED'].values[0]
        ## barriossincol is the list that conteocol is NaN
    barriossincol=conteodecolegios[conteodecolegios['conteo']==0]
    barriossincol['geometry']=barriossincol.apply(lambda row: just_suroccidente[just_suroccidente['BARRIO']==row.name]['geometry'].values[0], axis=1)


    barriossincol=gpd.GeoDataFrame(barriossincol, geometry='geometry')

    ## get geometry from 
    ## get lat and lon from barrios_sin_colegio
    barriossincol['lat']=barriossincol.apply(lambda row: row['geometry'].centroid.y, axis=1)
    barriossincol['lon']=barriossincol.apply(lambda row: row['geometry'].centroid.x, axis=1)
    barriossincol['nearcolegio']=''
    barriossincol["nearcolegio"]=barriossincol.apply(
    lambda row: find_neareast_COLE(row['lat'], row['lon']), 
    axis=1)

    st.header('Colegios que repiten cercania a barrios sin colegios')
    repe=barriossincol["nearcolegio"].value_counts()
    # index to column
    repe=repe.reset_index()
    # rename columns
    repe.rename(columns={'index':'colegio', 'nearcolegio':'conteo', 'BARRIO':'barrios'}, inplace=True)
    repe=pd.DataFrame(repe)
    gb=GridOptionsBuilder.from_dataframe(repe)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()

    gridOptions=gb.build()
    grid_response=AgGrid(repe, gridOptions=gridOptions,
                        height=400,
                        width='100%',
                        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                        fit_columns_on_grid_load=True,
                        allow_unsafe_jscode=True,
                        ## show index
                    
                        )
    repe=grid_response['data']
    selected=grid_response['selected_rows']
    selected=pd.DataFrame(selected)
    ## convert index to column in barriossincol
    barriossincolxd=barriossincol.drop(columns=['geometry'])

    barriossincol['distanciakm']=0
    ## remove NA 
    barriossincol=barriossincol.dropna()

    for i in range(len(barriossincol)):
        barriossincol['distanciakm'][i]=haversine(barriossincol['lat'][i], barriossincol['lon'][i], colegios_suroccidente[colegios_suroccidente['NOMBRE_SED']==barriossincol['nearcolegio'][i]]['lat'].values[0], colegios_suroccidente[colegios_suroccidente['NOMBRE_SED']==barriossincol['nearcolegio'][i]]['lon'].values[0])/1000
    
    ## convert index to column in barriossincol
    barriossincol=barriossincol.reset_index()
    barriossincol.rename(columns={'index':'BARRIO'}, inplace=True)
    barriossincol=pd.DataFrame(barriossincol)



    st.header('Distancia e IPM de los barrios sin colegios')
    #columns
    st.text(barriossincol.columns)
    gb=GridOptionsBuilder.from_dataframe(barriossincol[['BARRIO_left', 'distanciakm', 'ipm']])
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gridOptions=gb.build()
    grid_response=AgGrid(barriossincol[['BARRIO_left', 'distanciakm', 'ipm']], gridOptions=gridOptions,
                        height=400,
                        width='100%',
                        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                        fit_columns_on_grid_load=True,
                        allow_unsafe_jscode=True,
                        ## show index

                        )
    barriossincol=grid_response['data']
    selected=grid_response['selected_rows']
    selected=pd.DataFrame(selected)
    colegiooos=barriossincol
    
    st.write('Tomando en cuenta nuestra minimización, y a la vez los barrios antes priorizados, se procede a tomar como primer acercamiento de este caso, a los colegios:')
    st.write('1.	David Sanchez Juliao, siendo el segundo con mayor cercanía al Barrio priorizado de Cuchillas del Villate, se opta por el segundo mas cercano debido a la facilidad de acceso que tiene este colegio por la situación del barrio que se encuentra y sus alrededores. ')
    st.write('2.	INSTITUCION EDUCATIVA DISTRITAL SONIA AHUMADA, por su minimización de distancia con el barrio 7 de agosto. ')
    st.write('3.	INSTITUCION EDUCATIVA DISTRITAL BETSABE ESPINOSA, por su minimización de cercanías al barrio de el Golfo. ')
    st.write('Se quita de este análisis (de forma exploratoria), al barrio El bosque, ya que a pesar de ser aquel con mayor frecuencia de colegios, su gran amplitud no permite un acceso estratégico correcto en una primera instancia. ')
    st.write('Estos datos tomados de cercanía de instituciones serán una herramienta crucial al análisis exploratorio de las necesidades poblaciones de estos barrios, ya que un acercamiento a estos podría dar un muestreo de la situación, puesto que educadores y directivos, conocen de primera mano las dificultades que tiene el lugar, por su inmensa interacción que tiene con los jóvenes en su labor de educación. Mediante diálogos de Probarranquilla y la Fundación Santo Domingo, se determino que esta herramienta seria la mas eficaz por su amplitud de reconocimiento e información actualizada, siendo que su vez se elimina errores de medición por posibles incentivos perversos que existen al preguntarse directamente a los habitantes; a su vez ahorrándose costos de metodología extensiva. ')

    colegios_suroccidente=gpd.read_file('colegios.shp')
    colegios_suroccidente=colegios_suroccidente[['NOMBRE_SED', 'ACTIVIDAD']]
    ## reordenar columnas
    colegios_suroccidente=colegios_suroccidente[['NOMBRE_SED', 'ACTIVIDAD']]
    st.text(colegios_suroccidente.columns)
    result=colegios_suroccidente[colegios_suroccidente['NOMBRE_SED'].str.contains('BETSABE ESPINOSA', na=False)]
    # result=result.append(colegios_suroccidente[colegios_suroccidente['NOMBRE_SED'].str.contains('INSTITUCION EDUCATIVA DISTRITAL SONIA AHUMADA', na=False)])
    result=pd.concat([result, colegios_suroccidente[colegios_suroccidente['NOMBRE_SED'].str.contains('INSTITUCION EDUCATIVA DISTRITAL SONIA AHUMADA', na=False)]])
    # result=result.append(colegios_suroccidente[colegios_suroccidente['NOMBRE_SED'].str.contains('DAVID SANCHEZ', na=False)])
    result=pd.concat([result, colegios_suroccidente[colegios_suroccidente['NOMBRE_SED'].str.contains('DAVID SANCHEZ', na=False)]])
    #append is deprecated
    st.table(result)

    st.write('Todas 3 instituciones tienen educación hasta Básica Secundaria Media, lo que permite una amplitud de población al realizar encuestas a trabajadores de instituciones. ')

    st.write('Se recalca la necesidad, de hacer esto antes de entrar a encuestar a los educadores antes de entrar nuevamente los estudiantes, para que estos sean fáciles de allegar y/o no tengan distracciones que puedan hacer que la encuesta caiga en ruidos muestrales.   ')

    st.header('Análisis de activos empresariales en el sur occidente de Barranquilla')
    st.write('Para analizar el comportamiento de las poblaciones y como se distribuye las empresas de acuerdo a los factores que caracterizan el barrio, se toma en cuenta la base de datos de empresas registradas ante la cámara de comercio para 2021, para ello se hace una georreferenciación basada en la  dirección declarada en la base de datos; así se logra filtrar la cantidad de activos por CIIU y numero de empresas entre los diferentes barrios.')
    st.write('Distribucion total de suroccidente')


    BD2021=pd.read_csv('BD2021withcoordinates.csv')
    BD2021.head()
## create geometry column with lat and lon

    BD2021['geometry']=BD2021.apply(lambda row: Point(row['lon'],row['lat']),axis=1)
    BD2021=gpd.GeoDataFrame(BD2021,geometry='geometry')
    print(BD2021.columns)
    
    BD2021_suroccidente=gpd.sjoin(BD2021,just_suroccidente,how='inner',op='intersects')
    BD2021_suroccidente.head()
    BD2021_suroccidente_map=px.choropleth_mapbox(just_suroccidente, geojson=baqBARRIO.geometry, locations=just_suroccidente.index, color='ipm',
                                color_continuous_scale="Viridis",
                              ## auto range
                                range_color=(just_suroccidente['ipm'].min(), just_suroccidente['ipm'].max()),
                                mapbox_style="carto-positron",
                                ## barranquilla
                                zoom=11, center = {"lat": 10.988, "lon": -74.803},
                                opacity=0.8,
                             labels={'ipm':'IPM'},
                                title='Indice de pobreza multidimensional por barrio',
                                hover_name=just_suroccidente.index,
                                hover_data={'ipm':True}
                             )

    BD2021_suroccidente_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


## add scatter plot of BD2021 with color based on ACTIVOS 

    BD2021_suroccidente_map.add_trace(go.Scattermapbox(
        lat=BD2021_suroccidente['lat'],
        lon=BD2021_suroccidente['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(
           color=BD2021_suroccidente['ACTIVOS'],
            colorscale='Viridis',
            cmin=BD2021_suroccidente['ACTIVOS'].min(),
            cmax=BD2021_suroccidente['ACTIVOS'].quantile(0.85),
            colorbar=dict(
                title='ACTIVOS'
            )
        ),
        text=BD2021_suroccidente['RAZON_SOCIAL'],
    ))
    st.plotly_chart(BD2021_suroccidente_map)
    distribucion=BD2021_suroccidente.groupby('BARRIO').count()['RAZON_SOCIAL'].sort_values(ascending=True)
    distribucion=pd.DataFrame(distribucion)
    distribucion=distribucion.reset_index()
    distribucion.columns=['BARRIO','RAZON_SOCIAL']
    st.write('Se encuentra que aquellos barrios con mayores empresas son aquellos los cuales tienen un IPM menor, no obstante, esto será comprobado más adelante mediante métodos estadísticos descriptivos. El barrio del suroccidente que tiene mas empresas es La pradera con 280 registrados. ')


    
    ## ADD IPM COLUMN TO DISTRIBUTION
    distribucion=distribucion.merge(just_suroccidente, left_on='BARRIO', right_on='BARRIO')
    distribucion=distribucion[['BARRIO','RAZON_SOCIAL','ipm']]
    gb=GridOptionsBuilder.from_dataframe(distribucion[[ 'BARRIO','RAZON_SOCIAL', 'ipm']])
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gridOptions=gb.build()
    grid_response=AgGrid(distribucion[['BARRIO','RAZON_SOCIAL', 'ipm']],
                        height=400,
                        width='100%',
                        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                        fit_columns_on_grid_load=True,
                        allow_unsafe_jscode=True,
                        ## show index

                        )
    barriossincol=grid_response['data']
    distribucion_priorizados=distribucion[distribucion['BARRIO'].str.contains('7 DE AGOSTO', na=False)]
    # distribucion_priorizados=distribucion_priorizados.append(distribucion[distribucion['BARRIO'].str.contains('CUCHILLA DE VILLATE', na=False)])
    distribucion_priorizados=pd.concat([distribucion_priorizados,distribucion[distribucion['BARRIO'].str.contains('CUCHILLA DE VILLATE', na=False)]])
    # distribucion_priorizados=distribucion_priorizados.append(distribucion[distribucion['BARRIO'].str.contains('EL GOLFO', na=False)])
    distribucion_priorizados=pd.concat([distribucion_priorizados,distribucion[distribucion['BARRIO'].str.contains('EL GOLFO', na=False)]])
    st.table(distribucion_priorizados)

    st.write('Para los barrios priorizados, tal como era de esperarse, se encuentra una frecuencia de empresas mucho menor, esto puede deberse a temas de delincuencia que trae dificultad de acceso en la zona, lo que termina generando pocas oportunidades de empleo en la zona.  ')
    st.write('Un caso excepcional es EL Barrio El Golfo, que tiene diferentes CIIUs entre sus empresas sumando un total de 32, a pesar de su IPM alto, pero guardando lógica con lo anteriormente propuesto, que entre menor IPM mayor empresas se encontraran.   ')
    st.write('Si revisamos en detalle estas empresas, encontramos que se encuentran los siguientes CIIUs por cada uno de estos barrios:  ')

    CIIUs_7DEAGOSTO=BD2021_suroccidente[BD2021_suroccidente['BARRIO'].str.contains('7 DE AGOSTO', na=False)]
    CIIUs_CUCHILLADEVILLATE=BD2021_suroccidente[BD2021_suroccidente['BARRIO'].str.contains('CUCHILLA DE VILLATE', na=False)]
    CIIUs_ELGOLFO=BD2021_suroccidente[BD2021_suroccidente['BARRIO'].str.contains('EL GOLFO', na=False)]
   
    ## group by CIIU and count and sum ACTIVOS
    CIIUs_7DEAGOSTO=CIIUs_7DEAGOSTO.groupby('CIIU').agg({'RAZON_SOCIAL':'count', 'ACTIVOS':'sum'}).sort_values(by='ACTIVOS', ascending=False)
    CIIUs_7DEAGOSTO=pd.DataFrame(CIIUs_7DEAGOSTO)
    CIIUs_CUCHILLADEVILLATE=CIIUs_CUCHILLADEVILLATE.groupby('CIIU').agg({'RAZON_SOCIAL':'count', 'ACTIVOS':'sum'}).sort_values(by='ACTIVOS', ascending=False)
    CIIUs_CUCHILLADEVILLATE=pd.DataFrame(CIIUs_CUCHILLADEVILLATE)
    CIIUs_ELGOLFO=CIIUs_ELGOLFO.groupby('CIIU').agg({'RAZON_SOCIAL':'count', 'ACTIVOS':'sum'}).sort_values(by='ACTIVOS', ascending=False)
    CIIUs_ELGOLFO=pd.DataFrame(CIIUs_ELGOLFO)


    def get_RAZON_SOCIALswithCIIU(barrio):
        df_filter=st.multiselect('Seleccione el CIIU', BD2021_suroccidente[BD2021_suroccidente['BARRIO'].str.contains(barrio, na=False)]['CIIU'].unique())
        if len(df_filter)>0:
            df=BD2021_suroccidente[BD2021_suroccidente['BARRIO'].str.contains(barrio, na=False)]
            df=df[df['CIIU'].isin(df_filter)]
            df=df[['RAZON_SOCIAL','CIIU', 'ACTIVOS']]
            df=df.sort_values(by='ACTIVOS', ascending=False)
            st.table(df)
            return df
        else:
            pass
        

    st.header('CIIUs en 7 de Agosto')
    st.table(CIIUs_7DEAGOSTO)
    get_RAZON_SOCIALswithCIIU('7 DE AGOSTO')

    st.header('CIIUs en Cuchilla de Villate')
    st.table(CIIUs_CUCHILLADEVILLATE)
    get_RAZON_SOCIALswithCIIU('CUCHILLA DE VILLATE')

    st.header('CIIUs en El Golfo')
    st.table(CIIUs_ELGOLFO)
    get_RAZON_SOCIALswithCIIU('EL GOLFO')

    st.header('Equipamiento del barrio')
    st.header('Parques')
    parques=gpd.read_file('parques.shp')
    ## create coordinates
    parques['X']=parques['geometry'].x
    parques['Y']=parques['geometry'].y

    mapparques=px.choropleth_mapbox(just_suroccidente, geojson=just_suroccidente.geometry, locations=just_suroccidente.index, color='ipm',
                                color_continuous_scale="Viridis",
                                range_color=(ipm_baqbarrio_meanwithlocalidad['ipm'].min(), ipm_baqbarrio_meanwithlocalidad['ipm'].max()),
                                mapbox_style="carto-positron",
                                ## barranquilla
                                zoom=11, center = {"lat": 10.988, "lon": -74.803},
                                opacity=0.8,
                                labels={'ipm':'IPM', 'LOCALIDAD':'Localidad'},
                                title='Indice de pobreza multidimensional por barrio',
                                hover_name='BARRIO',
                                hover_data={'ipm':True, 'LOCALIDAD':True}
                                )
    mapparques.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    ## create geometry based on X and Y
    parques['geometry']=parques.apply(lambda x: Point(x['X'], x['Y']), axis=1)
    parques=gpd.GeoDataFrame(parques, geometry='geometry')

    ## ADD PARQUES points to map
    mapparques.add_trace(go.Scattermapbox(
        lat=parques['Y'],
        lon=parques['X'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=9
        ),
        text=parques['NOMBRE'],
    ))

    st.plotly_chart(mapparques)

    barrios_sinparques=just_suroccidente[~just_suroccidente['BARRIO'].isin(parques['BARRIO'])]
    barrios_sinparques=barrios_sinparques[['BARRIO', 'LOCALIDAD', 'ipm']]
    barrios_sinparques=barrios_sinparques.sort_values(by='ipm', ascending=False)
    gb=GridOptionsBuilder.from_dataframe(barrios_sinparques[[ 'BARRIO', 'LOCALIDAD', 'ipm']])
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gridOptions=gb.build()
    grid_response=AgGrid(barrios_sinparques[['BARRIO', 'LOCALIDAD', 'ipm']].sort_values(by='ipm',ascending=False), gridOptions=gridOptions,
                        height=400,
                        width='100%',
                        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                        fit_columns_on_grid_load=True,
                        allow_unsafe_jscode=True,
                        ## show index

                        )
    barriossincol=grid_response['data']
    st.write('Encontramos la problematica de que los barrios priorizados no tienen parques, un problema de interaccion social, que puede hacer incurrir el IPM por diferentes generaciones')
    st.write('Los parques mas cercanos a los barrios sin estos son:')

    ## nearest parque to barrio with no parque
    def get_nearest_parque(barrio):
        barrio=barrio
        barrio_point=just_suroccidente[just_suroccidente['BARRIO']==barrio]['geometry'].iloc[0]
        parques['distance']=parques.apply(lambda x: barrio_point.distance(x['geometry']), axis=1)
        return parques[parques['distance']==parques['distance'].min()]['NOMBRE'].iloc[0]

    def get_nearest_parque_distance(barrio):
        barrio=barrio
        barrio_point=just_suroccidente[just_suroccidente['BARRIO']==barrio]['geometry'].iloc[0]
        parques['distance']=parques.apply(lambda x: barrio_point.distance(x['geometry']), axis=1)
        return parques['distance'].min()
    


    barriossincol['nearest_parque']=barriossincol.apply(lambda x: get_nearest_parque(x['BARRIO']), axis=1)
    barriossincol['nearest_parque_distance']=barriossincol.apply(lambda x: get_nearest_parque_distance(x['BARRIO']), axis=1)
    barriossincol=barriossincol.sort_values(by='ipm', ascending=False)
    gb=GridOptionsBuilder.from_dataframe(barriossincol[[ 'BARRIO', 'LOCALIDAD', 'ipm', 'nearest_parque', 'nearest_parque_distance']])
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gridOptions=gb.build()
    grid_response=AgGrid(barriossincol[['BARRIO', 'LOCALIDAD', 'ipm', 'nearest_parque', 'nearest_parque_distance']].sort_values(by='ipm',ascending=False), gridOptions=gridOptions,
                        height=400, 
                        width='100%',
                        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                        fit_columns_on_grid_load=True,
                        allow_unsafe_jscode=True,
                        ## show index
    )

    st.header('Iglesias')
    iglesias=gpd.read_file('iglesias.shp')
    ## plot iglesias
    mapiglesias=px.choropleth_mapbox(just_suroccidente, geojson=just_suroccidente.geometry, locations=just_suroccidente.index, color='ipm',
                                color_continuous_scale="Viridis",                   
                                range_color=(ipm_baqbarrio_meanwithlocalidad['ipm'].min(), ipm_baqbarrio_meanwithlocalidad['ipm'].max()),
                                mapbox_style="carto-positron",
                                ## barranquilla
                                zoom=11, center = {"lat": 10.988, "lon": -74.803},
                                opacity=0.8,
                                labels={'ipm':'IPM', 'LOCALIDAD':'Localidad'},
                                title='Indice de pobreza multidimensional por barrio',
                                hover_name='BARRIO',
                                hover_data={'ipm':True, 'LOCALIDAD':True}
                                )
    mapiglesias.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    ## add iglesias
    ## create coordinates for iglesias based on geometry
    iglesias['X']=iglesias['geometry'].x
    iglesias['Y']=iglesias['geometry'].y
    mapiglesias.add_trace(go.Scattermapbox(
        lat=iglesias['Y'],
        lon=iglesias['X'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=9
        ),
        text=iglesias['PARROQUIA'],
    ))
    iglesias=gpd.sjoin(iglesias, just_suroccidente, how='left', op='intersects')
    
    st.plotly_chart(mapiglesias)
    ## barrios sin iglesias
    barrios_siniglesias=just_suroccidente[~just_suroccidente['BARRIO'].isin(iglesias['index_righ'])]
    barrios_siniglesias=barrios_siniglesias[['BARRIO', 'LOCALIDAD', 'ipm', 'geometry']]
    barrios_siniglesias=barrios_siniglesias.sort_values(by='ipm', ascending=False)
    gb=GridOptionsBuilder.from_dataframe(barrios_siniglesias[[ 'BARRIO', 'LOCALIDAD', 'ipm']])
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gridOptions=gb.build()
    grid_response=AgGrid(barrios_siniglesias[{'BARRIO', 'LOCALIDAD', 'ipm'}].sort_values(by='ipm',ascending=False), gridOptions=gridOptions,
                        height=400,                 
                        width='100%',
                        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                        fit_columns_on_grid_load=True,
                        allow_unsafe_jscode=True,
                        ## show index


    )

  
    
        ## calculate the nearest iglesia to each barrio without iglesia
    def nearest_iglesia(barrio, iglesias):
        return iglesias[iglesias.geometry.distance(barrio)==iglesias.geometry.distance(barrio).min()]['PARROQUIA'].values[0]

    barrios_siniglesias['nearest_iglesia']=barrios_siniglesias.apply(lambda x: nearest_iglesia(x['geometry'], iglesias), axis=1)
    st.write('Encontramos la problematica de que los barrios priorizados no tienen iglesias, un problema de interaccion social, que puede hacer incurrir el IPM por diferentes generaciones')
    st.write('Los barrios sin iglesias mas cercanos a las iglesias son:')
    st.write(barrios_siniglesias[['BARRIO', 'LOCALIDAD', 'ipm', 'nearest_iglesia']])
    parques=parques[{ 'BARRIO', 'NOMBRE'}]
    grupodecolegios=grupodecolegios[{ 'BARRIO_left', 'NOMBRE_SED'}]
    iglesias=iglesias[{ 'BARRIO', 'PARROQUIA'}]
    st.header('Revisiones por barrio')
    BD2021_suroccidente2=BD2021_suroccidente[['RAZON_SOCIAL', 'CIIU', 'BARRIO']]
    st.write('En esta seccion se cada una de las tematicas antes mencionadas. En el caso de no tener parques o iglesias, se muestra la mas cercana')
    st.header('7 DE AGOSTO')
    st.header('Parques:')
    if len(parques[parques['BARRIO']=='7 DE AGOSTO'])>0:
        st.write(parques[parques['BARRIO']=='7 DE AGOSTO'])
    else:
        ## get nearest parque
        nearest_parque=barriossincol[barriossincol['BARRIO']=='7 DE AGOSTO']['nearest_parque'].values[0]
        nearest_parque_distance=barriossincol[barriossincol['BARRIO']=='7 DE AGOSTO']['nearest_parque_distance'].values[0]
        st.write('El parque mas cercano es: '+nearest_parque+' a '+str(nearest_parque_distance)+' metros'+' que se encuentra en otro barrio')

    st.header('Iglesias:')
    if len(iglesias[iglesias['BARRIO']=='7 DE AGOSTO'])>0:
        st.write(iglesias[iglesias['BARRIO']=='7 DE AGOSTO'])
    else:
        ## get nearest parque
        nearest_iglesia=barrios_siniglesias[barrios_siniglesias['BARRIO']=='7 DE AGOSTO']['nearest_iglesia'].values[0]
        st.write('La iglesia mas cercana es: '+nearest_iglesia +', que se encuentra en otro barrio')
    st.header('Colegios:')
    # drop NAs in grupodecolegios
    grupodecolegios=grupodecolegios.dropna()
    if len(grupodecolegios[grupodecolegios['BARRIO_left']=='7 DE AGOSTO'])>0:
        st.write(grupodecolegios[grupodecolegios['BARRIO_left']=='7 DE AGOSTO'])
    else:
        nearest_colegio=colegiooos[colegiooos['BARRIO_left']=='7 DE AGOSTO']['nearcolegio'].values[0]
        nearest_colegio_distance=colegiooos[colegiooos['BARRIO_left']=='7 DE AGOSTO']['distanciakm'].values[0]
        st.write('El colegio mas cercano es: '+nearest_colegio+' a '+str(nearest_colegio_distance)+' metros'+', que se encuentra en otro barrio')
    st.write('Empresas:')
    st.write(BD2021_suroccidente2[BD2021_suroccidente2['BARRIO']=='7 DE AGOSTO'])
    st.header('EL GOLFO')
    st.header('Parques:')
    if len(parques[parques['BARRIO']=='EL GOLFO'])>0:
        st.write(parques[parques['BARRIO']=='EL GOLFO'])
    else:
        ## get nearest parque
        nearest_parque=barriossincol[barriossincol['BARRIO']=='EL GOLFO']['nearest_parque'].values[0]
        nearest_parque_distance=barriossincol[barriossincol['BARRIO']=='EL GOLFO']['nearest_parque_distance'].values[0]
        st.write('El parque mas cercano es: '+nearest_parque+' a '+str(nearest_parque_distance)+' metros'+' que se encuentra en otro barrio')

    st.header('Iglesias:')
    if len(iglesias[iglesias['BARRIO']=='EL GOLFO'])>0:
        st.write(iglesias[iglesias['BARRIO']=='EL GOLFO'])
    else:
        ## get nearest parque
        nearest_iglesia=barrios_siniglesias[barrios_siniglesias['BARRIO']=='EL GOLFO']['nearest_iglesia'].values[0]
        st.write('La iglesia mas cercana es: '+nearest_iglesia +', que se encuentra en otro barrio')
    st.header('Colegios:')
    # drop NAs in grupodecolegios
    grupodecolegios=grupodecolegios.dropna()
    if len(grupodecolegios[grupodecolegios['BARRIO_left']=='EL GOLFO'])>0:
        st.write(grupodecolegios[grupodecolegios['BARRIO_left']=='EL GOLFO'])
    else:
        nearest_colegio=colegiooos[colegiooos['BARRIO_left']=='EL GOLFO']['nearcolegio'].values[0]
        nearest_colegio_distance=colegiooos[colegiooos['BARRIO_left']=='EL GOLFO']['distanciakm'].values[0]
        st.write('El colegio mas cercano es: '+nearest_colegio+' a '+str(nearest_colegio_distance)+' metros'+', que se encuentra en otro barrio')
    st.write('Empresas:')
    st.write(BD2021_suroccidente2[BD2021_suroccidente2['BARRIO']=='EL GOLFO'])
    st.header('CUCHILLA DE VILLATE')
    st.header('Parques:')
    if len(parques[parques['BARRIO']=='CUCHILLA DE VILLATE'])>0:
        st.write(parques[parques['BARRIO']=='CUCHILLA DE VILLATE'])
    else:
        ## get nearest parque
        nearest_parque=barriossincol[barriossincol['BARRIO']=='CUCHILLA DE VILLATE']['nearest_parque'].values[0]
        nearest_parque_distance=barriossincol[barriossincol['BARRIO']=='CUCHILLA DE VILLATE']['nearest_parque_distance'].values[0]
        st.write('El parque mas cercano es: '+nearest_parque+' a '+str(nearest_parque_distance)+' metros'+' que se encuentra en otro barrio')

    st.header('Iglesias:')
    if len(iglesias[iglesias['BARRIO']=='CUCHILLA DE VILLATE'])>0:
        st.write(iglesias[iglesias['BARRIO']=='CUCHILLA DE VILLATE'])
    else:
        ## get nearest parque
        nearest_iglesia=barrios_siniglesias[barrios_siniglesias['BARRIO']=='CUCHILLA DE VILLATE']['nearest_iglesia'].values[0]
        st.write('La iglesia mas cercana es: '+nearest_iglesia +', que se encuentra en otro barrio')
    st.header('Colegios:')
    # drop NAs in grupodecolegios
    grupodecolegios=grupodecolegios.dropna()
    if len(grupodecolegios[grupodecolegios['BARRIO_left']=='CUCHILLA DE VILLATE'])>0:
        st.write(grupodecolegios[grupodecolegios['BARRIO_left']=='CUCHILLA DE VILLATE'])
    else:
        nearest_colegio=colegiooos[colegiooos['BARRIO_left']=='CUCHILLA DE VILLATE']['nearcolegio'].values[0]
        nearest_colegio_distance=colegiooos[colegiooos['BARRIO_left']=='CUCHILLA DE VILLATE']['distanciakm'].values[0]
        st.write('El colegio mas cercano es: '+nearest_colegio+' a '+str(nearest_colegio_distance)+' metros'+', que se encuentra en otro barrio')
    st.write('Empresas:')
    st.write(BD2021_suroccidente2[BD2021_suroccidente2['BARRIO']=='CUCHILLA DE VILLATE'])
    st.header('EL CARMEN')
    st.header('Parques:')
    if len(parques[parques['BARRIO']=='EL CARMEN'])>0:
        st.write(parques[parques['BARRIO']=='EL CARMEN'])
    else:
        ## get nearest parque
        nearest_parque=barriossincol[barriossincol['BARRIO']=='EL CARMEN']['nearest_parque'].values[0]
        nearest_parque_distance=barriossincol[barriossincol['BARRIO']=='EL CARMEN']['nearest_parque_distance'].values[0]
        st.write('El parque mas cercano es: '+nearest_parque+' a '+str(nearest_parque_distance)+' metros'+' que se encuentra en otro barrio')
    
    st.header('Iglesias:')
    if len(iglesias[iglesias['BARRIO']=='EL CARMEN'])>0:
        st.write(iglesias[iglesias['BARRIO']=='EL CARMEN'])
    else:
        ## get nearest parque
        nearest_iglesia=barrios_siniglesias[barrios_siniglesias['BARRIO']=='EL CARMEN']['nearest_iglesia'].values[0]
        st.write('La iglesia mas cercana es: '+nearest_iglesia +', que se encuentra en otro barrio')
    st.header('Colegios:')
    # drop NAs in grupodecolegios
    grupodecolegios=grupodecolegios.dropna()
    if len(grupodecolegios[grupodecolegios['BARRIO_left']=='EL CARMEN'])>0:
        st.write(grupodecolegios[grupodecolegios['BARRIO_left']=='EL CARMEN'])
    else:
        nearest_colegio=colegiooos[colegiooos['BARRIO_left']=='EL CARMEN']['nearcolegio'].values[0]
        nearest_colegio_distance=colegiooos[colegiooos['BARRIO_left']=='EL CARMEN']['distanciakm'].values[0]
        st.write('El colegio mas cercano es: '+nearest_colegio+' a '+str(nearest_colegio_distance)+' metros'+', que se encuentra en otro barrio')
    st.write('Empresas:')
    st.write(BD2021_suroccidente2[BD2021_suroccidente2['BARRIO']=='EL CARMEN'])
    st.header('ME QUEJO')
    st.header('Parques:')
    if len(parques[parques['BARRIO']=='ME QUEJO'])>0:
        st.write(parques[parques['BARRIO']=='ME QUEJO'])
    else:
        ## get nearest parque
        nearest_parque=barriossincol[barriossincol['BARRIO']=='ME QUEJO']['nearest_parque'].values[0]
        nearest_parque_distance=barriossincol[barriossincol['BARRIO']=='ME QUEJO']['nearest_parque_distance'].values[0]
        st.write('El parque mas cercano es: '+nearest_parque+' a '+str(nearest_parque_distance)+' metros'+' que se encuentra en otro barrio')

    st.header('Iglesias:')
    if len(iglesias[iglesias['BARRIO']=='ME QUEJO'])>0:
        st.write(iglesias[iglesias['BARRIO']=='ME QUEJO'])
    else:
        ## get nearest parque
        nearest_iglesia=barrios_siniglesias[barrios_siniglesias['BARRIO']=='ME QUEJO']['nearest_iglesia'].values[0]
        st.write('La iglesia mas cercana es: '+nearest_iglesia +', que se encuentra en otro barrio')
    st.header('Colegios:')
    # drop NAs in grupodecolegios
    grupodecolegios=grupodecolegios.dropna()
    if len(grupodecolegios[grupodecolegios['BARRIO_left']=='ME QUEJO'])>0:
        st.write(grupodecolegios[grupodecolegios['BARRIO_left']=='ME QUEJO'])
    else:
        nearest_colegio=colegiooos[colegiooos['BARRIO_left']=='ME QUEJO']['nearcolegio'].values[0]
        nearest_colegio_distance=colegiooos[colegiooos['BARRIO_left']=='ME QUEJO']['distanciakm'].values[0]
        st.write('El colegio mas cercano es: '+nearest_colegio+' a '+str(nearest_colegio_distance)+' metros'+', que se encuentra en otro barrio')
    st.write('Empresas:')
    st.write(BD2021_suroccidente2[BD2021_suroccidente2['BARRIO']=='ME QUEJO'])
    st.header('LA ESMERALDA')
    st.header('Parques:')
    if len(parques[parques['BARRIO']=='LA ESMERALDA'])>0:
        st.write(parques[parques['BARRIO']=='LA ESMERALDA'])
    else:
        ## get nearest parque
        nearest_parque=barriossincol[barriossincol['BARRIO']=='LA ESMERALDA']['nearest_parque'].values[0]
        nearest_parque_distance=barriossincol[barriossincol['BARRIO']=='LA ESMERALDA']['nearest_parque_distance'].values[0]
        st.write('El parque mas cercano es: '+nearest_parque+' a '+str(nearest_parque_distance)+' metros'+' que se encuentra en otro barrio')
    st.header('Iglesias:')
    if len(iglesias[iglesias['BARRIO']=='LA ESMERALDA'])>0:
        st.write(iglesias[iglesias['BARRIO']=='LA ESMERALDA'])
    else:
        ## get nearest parque
        nearest_iglesia=barrios_siniglesias[barrios_siniglesias['BARRIO']=='LA ESMERALDA']['nearest_iglesia'].values[0]
        st.write('La iglesia mas cercana es: '+nearest_iglesia +', que se encuentra en otro barrio')
    st.header('Colegios:')
    # drop NAs in grupodecolegios
    grupodecolegios=grupodecolegios.dropna()
    if len(grupodecolegios[grupodecolegios['BARRIO_left']=='LA ESMERALDA'])>0:
        st.write(grupodecolegios[grupodecolegios['BARRIO_left']=='LA ESMERALDA'])
    else:
        nearest_colegio=colegiooos[colegiooos['BARRIO_left']=='LA ESMERALDA']['nearcolegio'].values[0]
        nearest_colegio_distance=colegiooos[colegiooos['BARRIO_left']=='LA ESMERALDA']['distanciakm'].values[0]
        st.write('El colegio mas cercano es: '+nearest_colegio+' a '+str(nearest_colegio_distance)+' metros'+', que se encuentra en otro barrio')
    st.write('Empresas:')
    st.write(BD2021_suroccidente2[BD2021_suroccidente2['BARRIO']=='LA ESMERALDA'])
    st.header('EL CARMEN')
    st.header('Parques:')
    if len(parques[parques['BARRIO']=='EL CARMEN'])>0:
        st.write(parques[parques['BARRIO']=='EL CARMEN'])
    else:
        ## get nearest parque
        nearest_parque=barriossincol[barriossincol['BARRIO']=='EL CARMEN']['nearest_parque'].values[0]
        nearest_parque_distance=barriossincol[barriossincol['BARRIO']=='EL CARMEN']['nearest_parque_distance'].values[0]
        st.write('El parque mas cercano es: '+nearest_parque+' a '+str(nearest_parque_distance)+' metros'+' que se encuentra en otro barrio')

    st.header('Iglesias:')
    if len(iglesias[iglesias['BARRIO']=='EL CARMEN'])>0:
        st.write(iglesias[iglesias['BARRIO']=='EL CARMEN'])
    else:
        ## get nearest parque
        nearest_iglesia=barrios_siniglesias[barrios_siniglesias['BARRIO']=='EL CARMEN']['nearest_iglesia'].values[0]
        st.write('La iglesia mas cercana es: '+nearest_iglesia +', que se encuentra en otro barrio')
    st.header('Colegios:')
    # drop NAs in grupodecolegios
    grupodecolegios=grupodecolegios.dropna()
    if len(grupodecolegios[grupodecolegios['BARRIO_left']=='EL CARMEN'])>0:
        st.write(grupodecolegios[grupodecolegios['BARRIO_left']=='EL CARMEN'])
    else:
        nearest_colegio=colegiooos[colegiooos['BARRIO_left']=='EL CARMEN']['nearcolegio'].values[0]
        nearest_colegio_distance=colegiooos[colegiooos['BARRIO_left']=='EL CARMEN']['distanciakm'].values[0]
        st.write('El colegio mas cercano es: '+nearest_colegio+' a '+str(nearest_colegio_distance)+' metros'+', que se encuentra en otro barrio')
    st.write('Empresas:')
    st.write(BD2021_suroccidente2[BD2021_suroccidente2['BARRIO']=='EL CARMEN'])




        




if __name__=='__main__':
  app_layout()

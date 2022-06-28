from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
df = pd.read_csv('websites_after_changes.csv', header=0)
df_model = df.copy()
options = []
for col in df.columns[1:]:
     options.append({'label':'{}'.format(col, col), 'value':col})
df['all']=''
def generate_table(dataframe, max_rows=10, max_cols=5):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns[:max_cols+1]])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns[:max_cols+1]
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

df_opis = pd.DataFrame()
df_opis.loc[:,'mean'] = df.mean(numeric_only=True)
df_opis.loc[:,'median'] = df.median(numeric_only=True)
df_opis.loc[:,'min'] = df.min()
df_opis.loc[:,'max'] = df.max()
df_opis.loc[:,'skew'] = df.skew()
df_opis.loc[:,'std'] = df.std()
df_opis.reset_index(inplace=True)
df_opis.rename(columns={'index':'attribute'}, inplace=True)
app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='FakeNews Detection'),
    generate_table(df),

    html.Br(),
    html.H2(children='Analiza eksploracyjna'),
    html.H3(['Wizualizacja korelacji danych']),
    html.Label('Wybór X'),
    dcc.Dropdown(options=options, value = ' div_count', id='x_choice'),
    html.Label('Wybór Y'),
    dcc.Dropdown(options=options, value = 'exclamation_count', id='y_choice'),
    dcc.Graph(id='graph_corr'),
    html.Br(),
    html.H3(["Rozkład danych w zalezności od wartości 'label'"]),
    html.Label('Wybór cechy'),
    dcc.Dropdown(options=options[:-1], value=' div_count', id='violin_choice'),
    dcc.Graph(id='graph_violin'),
    html.Br(),
    html.H3(["Tabela danych opisujących rozkład atrybutów po normalizacji"]),
    dcc.Graph(id='graph_table'),
    html.Br(),
    html.H2(["Modelowanie"]),
    html.Label('Wybór metody uczenia'),
    dcc.Dropdown(options=['Regresja logistyczna', 'kNN'], value='kNN', id='model_choice'),
    dcc.Graph(id='graph_model_1'),
    dcc.Graph(id='graph_model_2')
])

@app.callback(
    Output('graph_corr', 'figure'),
    Output('graph_violin', 'figure'),
    Output('graph_table', 'figure'),
    Output('graph_model_1', 'figure'),
    Output('graph_model_2', 'figure'),
    Input('x_choice', 'value'),
    Input('y_choice', 'value'),
    Input('violin_choice', 'value'),
    Input('model_choice',  'value'))
def update_figure(x_choice, y_choice, violin_choice, model_choice):

    fig_corr = px.scatter(df, x=x_choice, y=y_choice)
    fig_corr.update_layout(transition_duration=500)
    fig_violin = go.Figure()
    fig_violin.add_trace(go.Violin(x=df['all'],
                            y=df[violin_choice][ df['label'] == 0 ],
                            legendgroup='Yes', scalegroup='Yes', name='Yes',
                            side='negative',
                            line_color='blue')
                )
    fig_violin.add_trace(go.Violin(x=df['all'],
                            y=df[violin_choice][ df['label'] == 1 ],
                            legendgroup='No', scalegroup='No', name='No',
                            side='positive',
                            line_color='orange')
                )
    fig_violin.update_traces(meanline_visible=True,width=80)
    fig_violin.update_layout(violingap=0, violinmode='overlay', xaxis_zeroline=False)

    fig_table = go.Figure(data=[go.Table(
        header=dict(values=list(df_opis.columns),
                fill_color='paleturquoise',
                align='left'),
        cells=dict(values=df_opis[1:].transpose().values.tolist(),
               fill_color='lavender',
               align='left'))
    ])
    X = df_model.iloc[:,:-1]
    y = df_model.iloc[:,-1]
    X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(X, y, test_size=0.3, random_state=5)
    if model_choice == 'kNN':
        k_range = range(1,200)
        scores_binary = {}
        scores_list_binary = []
        for k in k_range:
            knn_binary = KNeighborsClassifier(n_neighbors=k)
            knn_binary.fit(X_train_binary, y_train_binary)
            y_pred_binary = knn_binary.predict(X_test_binary)
            scores_binary[k] = metrics.accuracy_score(y_test_binary, y_pred_binary)
            scores_list_binary.append(metrics.accuracy_score(y_test_binary,y_pred_binary))
        fig_model_1 = go.Figure(go.Scatter(x=list(k_range), y=scores_list_binary))
        fig_model_1.update_layout(title="Dokładność dla róznych wartości 'k'", xaxis_title='Liczba sąsiadów', yaxis_title='Dokładność modelu')
        knn_binary = KNeighborsClassifier(n_neighbors=7)
        knn_binary.fit(X_train_binary, y_train_binary)
        y_pred_binary=knn_binary.predict(X_test_binary)
        confusion_matrix = metrics.confusion_matrix(y_test_binary, y_pred_binary)
        confusion_matrix = confusion_matrix.astype(int)

        layout = {
            "title": "Macierz pomyłek", 
            "xaxis": {"title": "Wartość przewidziana"}, 
            "yaxis": {"title": "Wartość prawdziwa"}
        }

        fig_model_2 = go.Figure(data=go.Heatmap(z=confusion_matrix,
                                        x=['0','1'],
                                        y=['0','1'],
                                        hoverongaps=False), layout=layout)
        fig_model_2.update_layout()
    else:
        clf_binary = LogisticRegression()
        clf_binary = clf_binary.fit(X_train_binary,y_train_binary)
        y_pred_binary = clf_binary.predict(X_test_binary)
        confusion_matrix = metrics.confusion_matrix(y_test_binary, y_pred_binary)
        confusion_matrix = confusion_matrix.astype(int)

        layout = {
            "title": "Macierz pomyłek", 
            "xaxis": {"title": "Wartość przewidziana"}, 
            "yaxis": {"title": "Wartość prawdziwa"}
        }

        fig_model_1 = go.Figure(data=go.Heatmap(z=confusion_matrix,
                                        x=['0','1'],
                                        y=['0','1'],
                                        hoverongaps=False), layout=layout)
        y_score = clf_binary.predict_proba(X_test_binary)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test_binary, y_score)
        df_thresh = pd.DataFrame({
            'False Positive Rate': fpr,
            'True Positive Rate': tpr
        }, index=thresholds)
        df_thresh.index.name = "Thresholds"
        df_thresh.columns.name = "Rate"

        fig_model_2 = px.line(
            df_thresh, title='Krzywa ROC',
            width=700, height=500
        )

        fig_model_2.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_model_2.update_xaxes(range=[0, 1], constrain='domain')
    return fig_corr, fig_violin, fig_table, fig_model_1, fig_model_2



if __name__ == '__main__':
    app.run_server(debug=True)
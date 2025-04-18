1. text

import streamlit as st

def main():
    st.title("This is Text Elements")
    st.header("This is Header")
    st.subheader("This is sub Header")
    st.markdown("This text is :red[colored red], and this is **:blue[colored] ** and bold.")
    st.write("-" * 50)
    st.markdown("""
    ### SubChater 1
     - :red[$\sqrt{x^2+y^2}=1$] is a Pythagorean identity. :pencil:
    """)
    st.write("-" * 50)
    st.markdown("## Chapter 1. \n"
                "- Streamlit is **_really_ cool**.\n"
                "   * This text is : blue[colored blue], and this is **:red[colored] ** and bold.")


if __name__ == "__main__":
    main()


2. data display dataframe

import streamlit as st
import seaborn as sns
import pandas as pd

@st.cache_data
def load_data():
    df = sns.load_dataset('iris')
    return df

def main():
    st.title("Data Display st.dataframe()")
    st.checkbox("Use container width", value=False, key = 'use_container_width')

    iris = load_data()
    st.dataframe(iris, use_container_width=st.session_state.use_container_width)

    # pandas style
    st.dataframe(iris.iloc[:5, 0:3].style.highlight_max(axis=1))

if __name__ == "__main__":
    main()


3. data display table metric

import streamlit as st
import seaborn as sns
import pandas as pd

def main():
    tips = sns.load_dataset('tips')
    tip_max = tips['tip'].max()
    tip_min = tips['tip'].min()

    st.metric(label="Max Tip", value=tip_max,
              delta=tip_max - tip_min)
    st.metric(label="Min Tip", value=tip_min,
              delta=tip_min - tip_max)
    st.table(tips.describe())

if __name__ == "__main__":
    main()


4. matplotlib

import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main():
    st.title("Streamlit with Matplotlib")
    tips = sns.load_dataset('tips')
    m_tips = tips.loc[tips['sex'] == 'Male']
    f_tips = tips.loc[tips['sex'] == 'Female']
    fig, ax = plt.subplots(ncols=2, figsize=(10, 6), sharex=True, sharey=True)
    ax[0].scatter(x = m_tips['total_bill'], y = m_tips['tip'])
    ax[0].set_title('Male')
    ax[1].scatter(x = f_tips['total_bill'], y = f_tips['tip'])
    ax[1].set_title('Female')
    fig.supxlabel('Total Bill($)')
    fig.supylabel('Tip($)')
    st.pyplot(fig)

if __name__ == "__main__":
    main()


5. seaborn

import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main():
    st.title("Streamlit with Seaborn")
    tips = sns.load_dataset('tips')
    m_tips = tips.loc[tips['sex'] == 'Male']
    f_tips = tips.loc[tips['sex'] == 'Female']
    fig, ax = plt.subplots(ncols=2, figsize=(10, 6), sharex=True, sharey=True)
    sns.scatterplot(data=m_tips, x = 'total_bill', y = 'tip', ax=ax[0])
    ax[0].set_title('Male')
    sns.scatterplot(data=f_tips, x = 'total_bill', y = 'tip', ax=ax[1])
    ax[0].set(xlabel=None, ylabel=None)
    ax[1].set_title('Female')
    ax[1].set(xlabel=None, ylabel=None)
    fig.supxlabel('Total Bill($)')
    fig.supylabel('Tip($)')
    st.pyplot(fig)

if __name__ == "__main__":
    main()


6. plotly

import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import seaborn as sns

def main():
    st.title("Streamlit with Plotly")
    tips = sns.load_dataset('tips')
    m_tips = tips.loc[tips['sex'] == 'Male']
    f_tips = tips.loc[tips['sex'] == 'Female']

    fig = make_subplots(rows = 1,
                        cols = 2,
                        subplot_titles=('Male', 'Female'),
                        shared_yaxes=True,
                        shared_xaxes=True,
                        x_title='Total Bill($)'
                        )
    fig.add_trace(go.Scatter(x = m_tips['total_bill'], y = m_tips['tip'], mode='markers'), row=1, col=1)
    fig.add_trace(go.Scatter(x = f_tips['total_bill'], y = f_tips['tip'], mode='markers'), row=1, col=2)
    fig.update_yaxes(title_text="Tip($)", row=1, col=1)
    fig.update_xaxes(range=[0, 60])
    fig.update_layout(showlegend=False)

    # Display visualization
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()



7. button

import streamlit as st
def calculate_sales_revenue(price, total_sales):
    revenue = price * total_sales
    return revenue

def main():
    st.title("Sales Revenue Calculator")

    price = st.slider("단가:", 1000, 10000, value=5000)
    total_sales = st.slider("전체 판매 갯수", 1, 1000, value=500)

    if st.button("매출액 계산"):
        revenue = calculate_sales_revenue(price, total_sales)
        st.write("전체 매출액은", revenue, "원(KRW)")


if __name__ == "__main__":
    main()


8. checkbox

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def main():
    st.title("Check Box Control")
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    show_plot = st.checkbox("시각화 보여주기")

    fig, ax = plt.subplots()
    ax.plot(x, y)

    if show_plot:
        st.pyplot(fig)

if __name__ == '__main__':
    main()    


9. raio

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# 데이터 불러오기
iris = sns.load_dataset('iris')

def plot_matplotlib():
    st.title('Scatter Plot with Matplotlib')
    fig, ax = plt.subplots()
    ax.scatter(iris['sepal_length'], iris['sepal_width'])
    st.pyplot(fig)

def plot_seaborn():
    st.title('Scatter Plot with Seaborn')
    fig, ax = plt.subplots()
    sns.scatterplot(iris, x = 'sepal_length', y = 'sepal_width')
    st.pyplot(fig)

def plot_plotly():
    st.title('Scatter Plot with Plotly')
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x = iris['sepal_length'],
                   y = iris['sepal_width'],
                   mode='markers')
    )
    st.plotly_chart(fig)

def main():
    st.title("Choose a plotting library")
    plot_type = st.radio(
        "어떤 스타일의 산점도를 보고 싶은가요?",
        ("Matplotlib", "Seaborn", "Plotly"))

    if plot_type == "Matplotlib":
        plot_matplotlib()
    elif plot_type == "Seaborn":
        plot_seaborn()
    elif plot_type == "Plotly":
        plot_plotly()

if __name__ == '__main__':
    main()


10. select & multiselect


import streamlit as st
import pandas as pd
import seaborn as sns

# 데이터 불러오기
iris = sns.load_dataset('iris')

def main():

    st.markdown("## Raw Data")
    st.dataframe(iris)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Select")
    column = st.selectbox("1개의 종을 선택하세요", iris.species.unique())
    st.dataframe(iris[iris['species']==column].reset_index())

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## MultiSelect")
    cols = st.multiselect("복수의 컬럼을 선택하세요", iris.columns)
    filtered_iris = iris.loc[:, cols]
    st.dataframe(filtered_iris)

if __name__ == "__main__":
    main()

    
11. slider_select_slider

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go

@st.cache_data
def load_data():
    # 데이터 불러오기
    tips = sns.load_dataset('tips')

    return tips

@st.cache_resource
def run_model(data, max_depth, min_samples_leaf):
    # 특성과 타겟 분리
    y = data['tip']
    X = data[['total_bill', 'size']]

    # 훈련, 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    st.write('선택된 max_depth:', max_depth, '& min_samples_leat:', min_samples_leaf)

    random_search = {'max_depth': [i for i in range(max_depth[0], max_depth[1])],
                     'min_samples_leaf': [min_samples_leaf]}

    clf = RandomForestRegressor()
    model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 10,
                                   cv = 4, verbose= 1, random_state= 101, n_jobs = -1)
    return model.fit(X_train,y_train), X_test, y_test

def prediction(model, X_test, y_test):
    # 예측
    y_test_pred = model.predict(X_test)

    # 성능 평가
    test_mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    return y_test_pred, test_mae, r2

def prediction_plot(X_test, y_test, y_test_pred, test_mae, r2):
    # 그래프 그리기
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=X_test['total_bill'], y=y_test, mode='markers', name='test', marker=dict(color='red'))
    )
    fig.add_trace(
        go.Scatter(x=X_test['total_bill'], y=y_test_pred, mode='markers', name='prediction', marker=dict(color='green'))
    )

    fig.update_layout(
        title='Tip Prediction with RandomForestRegressor',
        xaxis_title='Total Bill',
        yaxis_title='Total',
        annotations=[go.layout.Annotation(x=40, y=1.5,
                                            text=f'Test MAE: {test_mae:.3f}<br>R2 Score: {r2:.3f}',
                                            showarrow=False)]
    )

    st.plotly_chart(fig)

def main():
    # Hyperparameters
    max_depth = st.select_slider("Select max depth", options=[i for i in range(2, 30)], value=(5, 10))
    min_samples_leaf = st.slider("Minimum samples leaf", min_value=2, max_value=20)

    tips = load_data()
    model, X_test, y_test = run_model(tips, max_depth, min_samples_leaf)
    y_test_pred, test_mae, r2 = prediction(model, X_test, y_test)
    prediction_plot(X_test, y_test, y_test_pred, test_mae, r2)

if __name__ == "__main__":
    main()



12. number_date_input

import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import yfinance as yf


def main():
    st.sidebar.title("Stock Chart")

    ticker = st.sidebar.text_input("Enter a ticker (e.g. AAPL)", value="AAPL")
    st.sidebar.markdown('Tickers Link : [All Stock Symbols](https://stockanalysis.com/stocks/)')
    start_date = st.sidebar.date_input("Start date", value=pd.to_datetime('2023-01-01'))
    end_date = st.sidebar.date_input("End date", value=pd.to_datetime('today'))

    data = yf.download(ticker, start=start_date, end=end_date)
    data.columns = data.columns.droplevel(1)
    
    chart_type = st.sidebar.radio("Select chart type", ("Candlestick", "Line"))

    candlestick = go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])
    line = go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close')

    if chart_type == "Candlestick":
        fig = go.Figure(candlestick)
    elif chart_type == "Line":
        fig = go.Figure(line)
    else:
        pass

    fig.update_layout(title=f"{ticker} Stock {chart_type} Chart", xaxis_title="Date", yaxis_title="Price")

    # Plot the figure
    st.plotly_chart(fig)
    st.markdown("<hr>", unsafe_allow_html=True)

    num_row = st.sidebar.number_input('Number of Rows', min_value=1, max_value=len(data))
    st.dataframe(data[-num_row:].reset_index().sort_index(ascending=False).set_index('Date'), use_container_width=True)



if __name__ == "__main__":
    main()


13. sidebar


import streamlit as st
import matplotlib.pyplot as plt
def main():
    value1 = st.sidebar.slider('Select a Value object notation', 0, 100)
    st.sidebar.write(value1)

    with st.sidebar:
        value2 = st.slider('Select a Value with notation', 0, 100)
        st.write(value2)
    value3 = st.slider('Select a Value just slider, 0, 100')
    st.write(value3)

    with st.sidebar:
        st.markdown('### Matplotlib Added on Sidebar')
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        st.pyplot(fig)

if __name__ == "__main__":
    main()


14. layout


import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

def main():
    with st.sidebar:
        st.header("Sidebar")
        day = st.selectbox("Select a day", ["Thur", "Fri", "Sat", "Sun"])

    tips = sns.load_dataset("tips")
    filtered_tips = tips[tips["day"] == day]
    top_bill = filtered_tips["total_bill"].max()
    top_tip = filtered_tips["tip"].max()

    tab1, tab2, tab3 = st.sidebar.tabs(["Total Bill", "Tip", "Size"])

    with tab1:
        fig, ax = plt.subplots()
        st.header("Total Bill Amounts")
        sns.histplot(filtered_tips["total_bill"], kde=False, ax=ax)
        st.pyplot(fig)
    with tab2:
        fig, ax = plt.subplots()
        st.header("Tip Amounts")
        sns.histplot(filtered_tips["tip"], kde=False, ax=ax)
        st.pyplot(fig)
    with tab3:
        fig, ax = plt.subplots()
        st.header("Table Sizes")
        sns.boxplot(data=filtered_tips, x="sex", y="tip", ax=ax)
        st.pyplot(fig)

    container = st.container()
    col1, col2 = container.columns([1, 2])

    with col1:
        st.metric("Top Bill", f"${top_bill:.2f}")

    with col2:
        st.metric("Top Tip", f"${top_tip:.2f}")

    with container:
        with st.expander("Data Frame"):
            st.dataframe(filtered_tips)
        st.write("This is the footer1")

    st.write("This is the footer2")

    with container:
        st.write("This is the footer3")

    with container:
        st.empty()

    st.write("This is the footer4")


if __name__ == "__main__":
    main()



15. column_config


# -*- coding:utf-8 -*-
import streamlit as st
import seaborn as sns
import pandas as pd

def main():
    st.title("Column Configuration Demo")
    
    # Load tips dataset
    tips = sns.load_dataset('tips')
    
    tips_display = tips.copy()
    tips_display['tip_percent'] = (tips_display['tip'] / tips_display['total_bill'] * 100).round(1)
    
    # Configure the columns
    column_config = {
        'total_bill': st.column_config.NumberColumn(
            'Bill Amount',
            help='Total bill amount in dollars',
            format='$%.2f'
        ),
        'tip': st.column_config.NumberColumn(
            'Tip Amount',
            help='Tip amount in dollars', 
            format='$%.2f'
        ),
        'tip_percent': st.column_config.NumberColumn(
            'Tip %',
            help='Tip as percentage of bill',
            format='%.1f%%'
        ),
        'sex': st.column_config.SelectboxColumn(
            'Gender',
            help='Customer gender',
            width='small',
            options=['Male', 'Female']
        ),
        'smoker': st.column_config.CheckboxColumn(
            'Smoker',
            help='Whether the customer is a smoker'
        ),
        'day': st.column_config.SelectboxColumn(
            'Day',
            help='Day of the week',
            width='small',
            options=['Sun', 'Sat', 'Fri', 'Thur']
        ),
        'time': st.column_config.SelectboxColumn(
            'Time',
            help='Time of day',
            width='small',
            options=['Lunch', 'Dinner']
        ),
        'size': st.column_config.NumberColumn(
            'Party Size',
            help='Number of people in the dining party',
            min_value=1,
            max_value=6
        )
    }

    st.dataframe(
        tips_display,
        column_config=column_config,
        hide_index=True
    )

    st.markdown("### Daily Tips Data Visualization")
    
    # Create simple visualization dataframe with 4 days

    viz_data = pd.DataFrame({
        'day': ['Thur', 'Fri', 'Sat', 'Sun'],
        'area_data': [
            tips_display[tips_display['day'] == 'Thur']['total_bill'].tolist(),
            tips_display[tips_display['day'] == 'Fri']['total_bill'].tolist(),
            tips_display[tips_display['day'] == 'Sat']['total_bill'].tolist(),
            tips_display[tips_display['day'] == 'Sun']['total_bill'].tolist()
        ],
        'line_data': [
            tips_display[tips_display['day'] == 'Thur']['tip'].tolist(),
            tips_display[tips_display['day'] == 'Fri']['tip'].tolist(),
            tips_display[tips_display['day'] == 'Sat']['tip'].tolist(),
            tips_display[tips_display['day'] == 'Sun']['tip'].tolist()
        ],
        'bar_data': [
            tips_display[tips_display['day'] == 'Thur']['size'].tolist(),
            tips_display[tips_display['day'] == 'Fri']['size'].tolist(),
            tips_display[tips_display['day'] == 'Sat']['size'].tolist(),
            tips_display[tips_display['day'] == 'Sun']['size'].tolist()
        ]
    })
    # Set observed=False to retain current behavior for pandas operations
    st.dataframe(viz_data)
    pd.options.mode.copy_on_write = False

    # Calculate progress as percentage (0-100%)
    max_bill = max(tips_display['total_bill'])
    viz_data['progress'] = viz_data['area_data'].apply(lambda x: (sum(x)/len(x) / max_bill) * 100)

    # Configure visualization columns
    viz_config = {
        'area_data': st.column_config.AreaChartColumn(
            'Bills Distribution',
            y_min=0,
            y_max=max(tips_display['total_bill'])
        ),
        'line_data': st.column_config.LineChartColumn(
            'Tips Distribution',
            y_min=0,
            y_max=max(tips_display['tip'])
        ),
        'bar_data': st.column_config.BarChartColumn(
            'Party Sizes'
        ),
        'progress': st.column_config.ProgressColumn(
            'Avg Bill Progress',
            help='Average bill as percentage of maximum bill',
            format='%.1f%%',
            min_value=0,
            max_value=100
        )
    }

    st.dataframe(
        viz_data,
        column_config=viz_config,
        hide_index=True
    )
    

if __name__ == "__main__":
    main()
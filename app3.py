# changing to linear regression as better results

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mpl_dates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

# Load the datasets
df_occupation = pd.read_csv("jobdegrees-IT+FIELD.csv")
df_job_ads = pd.read_csv('ITopps_T.csv')
df_job_info = pd.read_csv('IToppsupdated-05-27extra.csv')
all_industry_df = pd.read_csv(r"C:\Users\kelli\PycharmProjects\WebscrapingSeek\Careers scraping\daily industry\allindustry-comb05-25.csv")

# Create the Streamlit app
# Sidebar options
sidebar_options = ["Degree Job Search", "Industry Trends"]
selected_option = st.sidebar.selectbox("Select Option", sidebar_options)

if selected_option == "Degree Job Search":
    # Code for Degree Job Search page
    st.title("Degree Job Search")

    # Dropdown for broad_field selection
    broad_fields = sorted(df_occupation['broad_field'].astype(str).unique())
    selected_broad_field = st.selectbox("Select Broad Field", broad_fields)

    # Filter dataset based on broad_field selection
    filtered_df = df_occupation[df_occupation['broad_field'] == selected_broad_field]

    if len(filtered_df) > 0:
        # Dropdown for degree selection
        degrees = sorted(filtered_df['degree'].astype(str).unique())
        selected_degree = st.selectbox("Select Degree", degrees)

        # Filter dataset based on degree selection
        filtered_df = filtered_df[filtered_df['degree'] == selected_degree]

        if len(filtered_df) > 0:
            # Dropdown for occupation selection
            occupations = sorted(filtered_df['occupation'].astype(str).unique())
            selected_occupation = st.selectbox("Select Occupation", occupations)

            # Check if selected occupation exists in the job ads dataset
            if selected_occupation in df_job_ads.columns:
                # Filter the job ads data for the selected occupation
                selected_job_ads = df_job_ads[[selected_occupation]]

                # Add a column for the number of the next day's data
                selected_job_ads['Next Day Jobs'] = selected_job_ads[selected_occupation].shift(-1)
                selected_job_ads.dropna(inplace=True)

                # saving test data dataframe
                test_data = selected_job_ads.iloc[-7:]

                def train_test_split(data):
                    data = data.values
                    n = 7
                    return data[:-n], data[-n:]

                lr_train_data, lr_test_data = train_test_split(selected_job_ads)

                X_train = lr_train_data[:, :-1]
                y_train = lr_train_data[:, -1]
                X_test = lr_test_data[:, :-1]
                y_test = lr_test_data[:, -1]

                # Create the XGB model
                model = LinearRegression()
                model.fit(X_train, y_train)

                lr_pred = model.predict(X_test)

                # Evaluate the model
                mse = mean_squared_error(y_test, lr_pred)
                r_squared = model.score(X_test, y_test)

                test_data['predictions'] = lr_pred

                # adding salary and qualification info
                job_info_row = df_job_info[df_job_info['name'] == selected_occupation]

                if not job_info_row.empty:
                    salary_text = job_info_row['salary'].values[0]
                    qualification_text = job_info_row['qualification'].values[0]

                    if pd.isnull(qualification_text):
                        qualification_text = "No info"

                    st.subheader("Average Salary:")
                    st.markdown(f"<h3 style='color:green'>{salary_text}</h3>", unsafe_allow_html=True)

                    st.subheader("Qualification requirements:")
                    st.write(qualification_text)

                else:
                    st.warning("No salary and qualification information available for the selected job.")

                # Create the line graph with original values and predictions
                st.subheader(f"Time Series of Job Posts for {selected_occupation}")
                plt.rcParams.update({'figure.figsize': (17, 3), 'figure.dpi': 300})
                fig, ax = plt.subplots()
                sns.lineplot(data=selected_job_ads, x=selected_job_ads.index, y='Next Day Jobs',
                             label='Original Values')
                sns.lineplot(data=test_data, x=test_data.index, y='predictions', color='red', label='Predictions')
                plt.grid(linestyle='-', linewidth=0.3)
                ax.tick_params(axis='x', rotation=90)

                # Adjusting x-axis date labels
                date_format = mpl_dates.DateFormatter('%Y-%m-%d')
                ax.xaxis.set_major_formatter(date_format)
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=len(selected_job_ads)))

                plt.legend()
                st.pyplot(fig)

                st.subheader("Model Accuracy:")
                st.write("Mean Squared Error:", mse)
                st.write("R-squared:", r_squared)

                st.subheader("Test Data with Predictions dataframe:")
                st.dataframe(test_data)

            else:
                st.warning("Selected job is not available in the dataset.")

elif selected_option == "Industry Trends":
    st.header("Industry Trends")

    # Create the dropdown menu for selecting the industry
    selected_industry = st.selectbox("Select an industry", all_industry_df["industry"].unique())

    # Filter the data for the selected industry
    selected_industry_data = all_industry_df[all_industry_df["industry"] == selected_industry]

    # Extract the time series data and dates
    time_series_data = selected_industry_data.iloc[:, 1:].values.flatten()
    dates = selected_industry_data.columns[1:]

    # Remove NaN values from time series data and corresponding dates
    valid_data_mask = ~pd.isnull(time_series_data)
    time_series_data = time_series_data[valid_data_mask]
    dates = dates[valid_data_mask]

    # Create a line graph for the time series trend
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=time_series_data, mode='lines', name='Job Postings'))
    fig.update_layout(xaxis_title='Date', yaxis_title='Job Postings', title=f"Time Series Trend for {selected_industry}")
    st.plotly_chart(fig)

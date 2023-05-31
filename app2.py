# ## WORKING STREAMLIT APP WITH MODEL.
#
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mpl_dates
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the first dataset
df_occupation = pd.read_csv("jobdegrees-IT+FIELD.csv")

# Load the second dataset
df_job_ads = pd.read_csv('ITopps_T.csv')
df_job_ads['Date'] = pd.to_datetime(df_job_ads['Date'], format='%d/%m/%Y')   # Convert Date column to datetime type
df_job_ads.set_index('Date', inplace=True)  # Set Date column as the index

# Create the Streamlit app
def main():
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

                xgb_train_data, xgb_test_data = train_test_split(selected_job_ads)

                X_train = xgb_train_data[:, :-1]
                y_train = xgb_train_data[:, -1]
                X_test = xgb_test_data[:, :-1]
                y_test = xgb_test_data[:, -1]

                # Create the XGB model
                model = XGBRegressor(objective="reg:squarederror", n_estimators=500)
                model.fit(X_train, y_train)

                xgb_pred = model.predict(X_test)

                # Evaluate the model
                mse = mean_squared_error(y_test, xgb_pred)
                r_squared = model.score(X_test, y_test)

                st.write("Mean Squared Error:", mse)
                st.write("R-squared:", r_squared)

                test_data['predictions'] = xgb_pred

                # Shift the index of test_data by 1 day
                test_data.index = test_data.index + pd.DateOffset(days=1)

                # Create a line graph with original values and predictions
                plt.rcParams.update({'figure.figsize': (17, 3), 'figure.dpi': 300})
                fig, ax = plt.subplots()
                sns.lineplot(data=selected_job_ads, x=selected_job_ads.index, y='Next Day Jobs',
                             label='Original Values')
                sns.lineplot(data=test_data, x=test_data.index, y='predictions', color='red', label='Predictions')
                plt.grid(linestyle='-', linewidth=0.3)
                ax.tick_params(axis='x', rotation=90)

                # Adjusting x-axis date labels
                date_format = mpl_dates.DateFormatter('%Y-%m-%d')  # Customize the date format as per your data
                ax.xaxis.set_major_formatter(date_format)
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=len(selected_job_ads)))

                plt.legend()
                st.pyplot(fig)

                # Display the prediction dataframe
                st.subheader("Test Data with Predictions:")
                st.dataframe(test_data)

            else:
                st.write("Selected occupation is not available in the dataset.")


if __name__ == '__main__':
    main()

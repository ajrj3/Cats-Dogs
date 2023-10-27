import numpy as np

from catsdogs import data


def test_preprocess_img():
    arr = 255*np.ones((5, 5, 3))
    ones = np.ones((5, 5, 3))
    arr_scaled = data.preprocess_img(arr)
    assert np.array_equal(ones, arr_scaled)
    
def test_dataset(df):
    """Test dataset quality and integrity"""
    column_list = ['Filename', 'Image', 'Cat']
    df.expect_table_columns_to_match_ordered_list(column_list=column_list)
    df.expect_column_values_to_be_of_type(column='Filename', type_='str')
    df.expect_column_distinct_values_to_equal_set(column='Cat', value_set=[0,1])
    assert isinstance(df['Image'].iloc[0], np.ndarray)
    assert df['Image'].iloc[0].shape == (200, 200, 3)

def test_to_numpyarray(df):
    arr = data.to_numpyarray(df['Image'])
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (20, 200, 200, 3)




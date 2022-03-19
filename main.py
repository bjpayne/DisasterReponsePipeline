import sys
from DisasterResponse import DisasterResponse

if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset

    categories = sys.argv[0]

    disasterResponse = DisasterResponse(data_file, categories)

    df = disasterResponse.extract_data()

    df = disasterResponse.transform_data(df)

import pandas as pd


class WriteGenerateData:

    @staticmethod
    def writeGenerateDataToH5(filename, dfHeader, dfData):
        """

        :param filename: the name of the file in which the data will be entered
        :param dfHeader:
        :param dfData:
        :return:
        """
        hdf = pd.HDFStore(filename)
        hdf.put('header', dfHeader)
        hdf.put('data', dfData)
        hdf.close()

    # @staticmethod
    # def fprintf(stream, format_spec, *args):
    #     """
    #     Function to format the string and write in the file
    #     :param stream: the file
    #     :param format_spec: format of the string which write
    #            Ex: 'c%s;id%u;'
    #     :param args: string to add in the file
    #     :return: No return -> add the line to the file
    #     """
    #     stream.write(format_spec % args)

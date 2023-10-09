import argparse
import os
import glob
import subprocess
import logging



def funtion_parse_cpg(c_in_folder, cpg_out_file, path_joern, countFile):
    file = os.listdir(c_in_folder)
    cmd_parse = ''
    cmd_cd = ''

    if (file == None):
        logging.debug('Folder is empty !')
    else:
        for c in file:
            file_num = c.split('_')[1].split('.')[0]
            countFile += 1
            if file_num == '1':
                cmd_parse = 'joern-parse.bat -o ' + cpg_out_file + str(countFile) + '_1.bin' + ' ' + c_in_folder + str(
                    countFile) + '_1.c' + ' --language c'
                cmd_cd = f'cd /d "{path_joern}"'
            elif file_num == '0':
                cmd_parse = 'joern-parse.bat -o ' + cpg_out_file + str(countFile) + '_0.bin' + ' ' + c_in_folder + str(
                    countFile) + '_0.c' + ' --language c'
                cmd_cd = f'cd /d "{path_joern}"'

            logging.debug(cmd_parse + ' output in ' + cpg_out_file)
            subprocess.call(f'{cmd_cd} && {cmd_parse}', shell=True)
        pass

def funtion_cpg_to_csv(folder_out_csv, folder_out_cpg, countFile, path_joern):
    cmd_to_csv = ''
    cmd_cd = ''
    cpg_out_file = os.listdir(folder_out_cpg)

    if cpg_out_file == None:
        logging.debug('Folder is empty !')

    else:
        for c in cpg_out_file:
            countFile += 1
            file_num = c.split('_')[1].split('.')[0]
            if file_num == '1':
                cmd_to_csv = 'joern-export.bat' + ' --repr=all' + ' --format=neo4jcsv' + ' --out ' + folder_out_csv + str(
                    countFile) + '_1 ' + folder_out_cpg + str(countFile) + '_1.bin'
                cmd_cd = f'cd/d "{path_joern}"'
            elif file_num == '0':
                cmd_to_csv = 'joern-export.bat' + ' --repr=all' + ' --format=neo4jcsv' + ' --out ' + folder_out_csv + str(
                    countFile) + '_0 ' + folder_out_cpg + str(countFile) + '_0.bin'
                cmd_cd = f'cd/d "{path_joern}"'
            logging.debug(cmd_to_csv + ' output to ' + folder_out_csv + c)
            subprocess.call(f'{cmd_cd} && {cmd_to_csv}', shell=True)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2cpg", help='Type of parse ', default="c2cpg")
    parser.add_argument("--cpg2csv", help='Type of parse ', default='cpg2csv')
    args = parser.parse_args()
    if args.c2cpg == "c2cpg":
        parser.add_argument("--c", help='To cpg from c', default="../vuldet/data/")
        parser.add_argument("--cpg_out", help='Folder for cpg output', default="../data/cpg/")
        parser.add_argument("--start", help='Start of position', default=0)
        parser.add_argument("--path_joern", help='Directory containing joern-cli',
                                    default="../vuldet/Vulder/joern/joern/joern-cli/")
        args = parser.parse_args()
    print(args)

    logging.basicConfig(level=logging.DEBUG, filename='runtime_cpg.log', filemode='w', format=(
            '%(levelname)s:\t'
            '%(filename)s:'
            '%(funcName)s():'
            '%(lineno)d\t'
            '%(message)s'
        )
                            )
    # funtion_parse_cpg(args.c, args.cpg_out, args.path_joern, args.start)
    #
    # if args.cpg2csv:
    #     parser.add_argument("--cpg", help='To csv from cpg', default="../data/cpg")
    #     parser.add_argument("--csv_out", help='Folder for cpg output', default="../vuldet/Vulder/data/data_csv/")
    #     parser.add_argument("--start", help='Start of position', default=0)
    #     parser.add_argument("--path_joern", help='Directory containing joern-cli',
    #                         default="../vuldet/Vulder/joern/joern/joern-cli/")
    #
    #     logging.basicConfig(level=logging.DEBUG, filename='runtime_export_csv.log', filemode='w', format=(
    #         '%(levelname)s:\t'
    #         '%(filename)s:'
    #         '%(funcName)s():'
    #         '%(lineno)d\t'
    #         '%(message)s'
    #     )
    #                         )
    #     funtion_cpg_to_csv(args.cpg_out, args.csv_out, args.start, args.path_joern)

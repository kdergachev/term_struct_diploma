# term_struct_diploma
This repo contains most files involved in practical part of my diploma as well as the text itself. The general overview of the work is using 2 years of weekly US Treasury data different methods to fitting restricted for multicollinearity Nelson-Siegel-Svensson model were compared. More in-depth description is in [diploma_text.pdf](diploma_text.pdf).


The data used was obtained using Bloomberg Terminal. In data_mod one can see the result of modifying the tables colleced using clear_xl function from table_prep file. Naming is done according to YYMMDD format to work without additional sorting. 


In the results folder there are multiple folders with different attempts at optimizing the data, the ones used and the most correct are the ones named \*\_points_price_NSS_w_double. Structure of these folders is as follows: there is a subfolder for each day (YYMMDD) containing starting points used in that day's optimization and the best Theta obtained from optimizations as well as one .csv file with all results in a tabular form.


In scripts_dipl there are the python scripts used. main.py basically uses functions in optimization.py and table_prep() and runs the main function get_results() while outfiles.py is used for tests and to generate plots/tables to be inserted in the paper. customized\_packages contains optimization algorithms I had to modify to better suit my needs. Everything after if __name__ == "main" is just leftovers of tests and not relevant same goes for tests.py. Scripts for the most part (but not completely) can be executed to get results, but there are still parts left from me executing parts of code in chunks while it was not complete. Main modifications in customized_packages are additions of breaker class objects and stopping argument to get results of fast optimization.


used_in_paper contains plots and tables (not yet combined) that were added as figures generated in outfiles.py.

In software\_versions_and_modifications.txt packages used and their versions as well as what was modified is outlined.


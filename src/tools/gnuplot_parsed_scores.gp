# Plots data. Run:
#   ~$ gnuplot -e "parsefile='myfile.dat'" gnuplot_parsed_scores.gp
#   ~$ epspdf parse_scores_plot.eps

set terminal postscript eps enhanced mono "Helvetica" 20
set encoding iso_8859_1
set out 'parse_scores_plot.eps'

set grid
set key right bottom
#set style line 1 lc rgb '#aa0000' lt 1 lw 2 pt 4 ps 1.5 pi 5
set style line 2 lc rgb '#00aa00' lt 1 lw 2 pt 5 ps 1.5 pi 5
set style line 3 lc rgb '#00aa00' lt 2 lw 2 pt 7 ps 1.5 pi 5
set style line 4 lc rgb '#00aa00' lt 3 lw 2 pt 9 ps 1.5 pi 5
plot parsefile every 5 with linespoints ls 2 t "net-2"


python train_regression.py --epochs 300 --config regress_genTops_lin_ctWRe --prefix v3_0p4_1010 --dRN 0.4 --conv_params "( (0.0, [10, 10]), )" --readout_params "(0.0, [32, 32])"
python train_regression.py --epochs 300 --config regress_genTops_lin_ctWRe --prefix v3_0p4_2020 --dRN 0.4 --conv_params "( (0.0, [20, 20]), )" --readout_params "(0.0, [32, 32])"
python train_regression.py --epochs 300 --config regress_genTops_lin_ctWRe --prefix v3_0p4_1010_1010 --dRN 0.4 --conv_params "( (0.0, [10, 10]), (0.0, [10, 10]) )" --readout_params "(0.0, [32, 32])"
python train_regression.py --epochs 300 --config regress_genTops_lin_ctWRe --prefix v3_0p4_1010_2020 --dRN 0.4 --conv_params "( (0.0, [10, 10]), (0.0, [20, 20]) )" --readout_params "(0.0, [32, 32])"
python train_regression.py --epochs 300 --config regress_genTops_lin_ctWRe --prefix v3_0p4_2020_2020 --dRN 0.4 --conv_params "( (0.0, [20, 20]), (0.0, [20, 20]) )" --readout_params "(0.0, [32, 32])"

python train_regression.py --epochs 300 --config regress_genTops_lin_ctWRe --prefix v3_0p3_1010 --dRN 0.3 --conv_params "( (0.0, [10, 10]), )" --readout_params "(0.0, [32, 32])"
python train_regression.py --epochs 300 --config regress_genTops_lin_ctWRe --prefix v3_0p3_2020 --dRN 0.3 --conv_params "( (0.0, [20, 20]), )" --readout_params "(0.0, [32, 32])"
python train_regression.py --epochs 300 --config regress_genTops_lin_ctWRe --prefix v3_0p3_1010_1010 --dRN 0.3 --conv_params "( (0.0, [10, 10]), (0.0, [10, 10]) )" --readout_params "(0.0, [32, 32])"
python train_regression.py --epochs 300 --config regress_genTops_lin_ctWRe --prefix v3_0p3_1010_2020 --dRN 0.3 --conv_params "( (0.0, [10, 10]), (0.0, [20, 20]) )" --readout_params "(0.0, [32, 32])"
python train_regression.py --epochs 300 --config regress_genTops_lin_ctWRe --prefix v3_0p3_2020_2020 --dRN 0.3 --conv_params "( (0.0, [20, 20]), (0.0, [20, 20]) )" --readout_params "(0.0, [32, 32])"

#python train_regression.py --config regress_genTops_lin_ctWRe --prefix v3_0p5_1010 --dRN 0.5 --conv_params "( (0.0, [10, 10]), )" --readout_params "(0.0, [32, 32])"
#python train_regression.py --config regress_genTops_lin_ctWRe --prefix v3_0p5_2020 --dRN 0.5 --conv_params "( (0.0, [20, 20]), )" --readout_params "(0.0, [32, 32])"
#python train_regression.py --config regress_genTops_lin_ctWRe --prefix v3_0p5_1010 --dRN 0.5 --conv_params "( (0.0, [10, 10]), (0.0, [10, 10]) )" --readout_params "(0.0, [32, 32])"
#python train_regression.py --config regress_genTops_lin_ctWRe --prefix v3_0p5_1020 --dRN 0.5 --conv_params "( (0.0, [10, 10]), (0.0, [20, 20]) )" --readout_params "(0.0, [32, 32])"
#python train_regression.py --config regress_genTops_lin_ctWRe --prefix v3_0p5_2020 --dRN 0.5 --conv_params "( (0.0, [20, 20]), (0.0, [20, 20]) )" --readout_params "(0.0, [32, 32])"

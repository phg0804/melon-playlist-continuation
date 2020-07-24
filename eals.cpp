// elementwise ALS
#include <iostream>
#include <string>
#include <time.h>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include "jsoncpp/json.h"
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace Eigen;
using namespace Json;

typedef Matrix<double, Dynamic, Dynamic> MatrixXd;

class Info {
public:
    int user, item;
    double score;
};

class Element {
public:
    double val;
    int ind;
    Element *dual;
};

class MatrixMM {
public:
    vector<Info> data;
    int row, col, N, prev_N;
    MatrixMM() {
        row = col = N = prev_N = 0;
    }

    void read_mm(string fname) {
        /*
         * matrix market format 
         * first line : row number, col number, nonzero element number
         * second~ line : row, col, score
         */
        int r, c, n;
        FILE *in = fopen(fname.c_str(), "r");
        fscanf(in,"%d%d%d",&r, &c, &n);
        row += r, prev_N = N, N += n;
        col = c;
        data.resize(N);
        for(int i=prev_N;i<N;i++) {
            fscanf(in,"%d%d%lf",&data[i].user, &data[i].item, &data[i].score);
        }
    }

    void fix_index(int train_user) {
        for(int i=prev_N;i<N;i++) {
            data[i].user += train_user;
        }
    }
};

class eALS {
public:
    eALS(int _num_factor, int _max_epoch, double _lambda, 
            double _confidence, double _diff_threshold, bool _verbose, 
            string _fname_train, string _fname_val):
        num_factor(_num_factor), max_epoch(_max_epoch), lambda(_lambda), 
        confidence(_confidence), diff_threshold(_diff_threshold), verbose(_verbose), 
        assigned(false), fname_train(_fname_train), fname_val(_fname_val) {
        }

    int num_factor, topn, max_epoch, num_user, num_item, val_user;
    double lambda, confidence, diff_threshold;
    bool verbose, assigned;
    string fname_train, fname_val;
    MatrixXd P, Q, P_current, Q_current;
    MatrixMM mmdata, valdata;
    vector<int> col_num, row_num, mask_validate_user;
    vector<vector<int>> ground_truth;
    vector<vector<Element>> rui_hat, riu_hat;

    void loadData() {
        mmdata.read_mm(fname_train);
        int train_user = mmdata.row;
        mmdata.read_mm(fname_val);
        mmdata.fix_index(train_user);
        val_user = mmdata.row - train_user;
        num_user = mmdata.row;
        num_item = mmdata.col;
        printf("Matrix density : %lf\n", (double)mmdata.N / num_user / num_item);
        printf("N : %d\n", mmdata.N);
        printf("# of users : %d, # of items : %d\n", num_user, num_item);
        fflush(stdout);
    }

    void train() {
        P = MatrixXd::Random(num_user, num_factor); 
        P_current.resize(num_user, num_factor); 
        Q = MatrixXd::Random(num_item, num_factor); 
        Q_current.resize(num_item, num_factor);
        getRhat();
        for(int iter=0;iter<max_epoch;iter++) {
            printf("Epoch %3d start\n", iter+1);
            fflush(stdout);
            train_epoch();
            double diff = checkConvergence();
            if(verbose) {
                if(diff<diff_threshold) {
                    printf("\nEpoch : %d\n", iter + 1);
                    printf("P and Q converged.\n");
                    break;
                }
                if(iter==max_epoch-1) {
                    printf("\nEpoch : %d\n", iter + 1);
                    printf("Reached maximum iteration.\n");
                }
            }
        }
    }

    void getRhat() {
        int r, c;
        rui_hat.assign(P.rows(), vector<Element>());
        col_num.assign(P.rows(), 0);
        riu_hat.assign(Q.rows(), vector<Element>());
        row_num.assign(Q.rows(), 0);
        for(int i=0;i<mmdata.N;i++) {
            r = mmdata.data[i].user;
            c = mmdata.data[i].item;
            rui_hat[r].push_back({P.row(r).dot(Q.row(c)), c, NULL});
            riu_hat[c].push_back({rui_hat[r][col_num[r]].val, r, NULL});
            col_num[r]++;
            row_num[c]++;
        }
        col_num.assign(P.rows(), 0);
        row_num.assign(Q.rows(), 0);
        for(int i=0;i<mmdata.N;i++) {
            r = mmdata.data[i].user;
            c = mmdata.data[i].item;
            rui_hat[r][col_num[r]].dual = &riu_hat[c][row_num[c]];
            riu_hat[c][row_num[c]].dual = &rui_hat[r][col_num[r]];
            col_num[r]++;
            row_num[c]++;
        }
    }

    void train_epoch() {
        //clock_t starttime;
        double puf, qif, starttime;

        // train user factors
        //starttime = clock();
        /*
        starttime = omp_get_wtime();
        MatrixXd Sq(num_factor, num_factor);
        Sq = Q.transpose() * Q;
        #pragma omp parallel
		{
			#pragma omp for
			for(int u=0;u<P.rows();u++) {
				double numerator = 0, denominator = 0;
				vector<double> rui_hat_f, riu_hat_f;
				for(int f=0;f<num_factor;f++) {
					rui_hat_f.resize(col_num[u]);
					numerator = P(u, f) * Sq(f, f) - P.row(u).dot(Sq.row(f));
					denominator = Sq(f, f) + lambda;
					for(int i_ind=0;i_ind<col_num[u];i_ind++) {
						qif = Q(rui_hat[u][i_ind].ind, f);
						rui_hat_f[i_ind] = rui_hat[u][i_ind].val - P(u, f) * qif;
						numerator += (1 + confidence - confidence * rui_hat_f[i_ind]) * qif;
						denominator += confidence * qif * qif;
					}
					P(u, f) = numerator / denominator;
					for(int i_ind=0;i_ind<col_num[u];i_ind++) {
						qif = Q(rui_hat[u][i_ind].ind, f);
						rui_hat[u][i_ind].dual->val = rui_hat[u][i_ind].val \
													= P(u, f) * qif + rui_hat_f[i_ind]; 
					}
				}
			}
		}
        printf("Updating P took ");
        printf("%.2lfs.\n", omp_get_wtime() - starttime);
        fflush(stdout);
        */
        FILE *fp = fopen("user.txt", "r");
        for(int u=0;u<P.rows();u++) {
        	for(int i=0;i<100;i++) {
        		float x;
        		fscanf(fp, "%f", &x);
        		P(u,i) = x;
			}
		}
		cout << P.row(0) << endl;
		fclose(fp);

        // train item factors
        //starttime = clock();
        starttime = omp_get_wtime();
        MatrixXd Sp(num_factor, num_factor);
        Sp = P.transpose() * P;
        #pragma omp parallel
		{
			#pragma omp for
			for(int i=0;i<Q.rows();i++) {
				double numerator = 0, denominator = 0;
				vector<double> rui_hat_f, riu_hat_f;
				for(int f=0;f<num_factor;f++) {
					riu_hat_f.resize(row_num[i]);
					numerator = Q(i, f) * Sp(f, f) - Q.row(i).dot(Sp.row(f));
					denominator = Sp(f, f) + lambda;
					for(int u_ind=0; u_ind<row_num[i]; u_ind++) {
						puf = P(riu_hat[i][u_ind].ind, f);
						riu_hat_f[u_ind] = riu_hat[i][u_ind].val - Q(i, f) * puf;
						numerator += (1 + confidence - confidence * riu_hat_f[u_ind]) * puf;
						denominator += confidence * puf * puf;
					}
					Q(i, f) = numerator/denominator;
					for(int u_ind=0;u_ind<row_num[i];u_ind++) {
						puf = P(riu_hat[i][u_ind].ind, f);
						riu_hat[i][u_ind].dual->val = riu_hat[i][u_ind].val \
													= riu_hat_f[u_ind] + Q(i, f) * puf;
					}
				}
			}
		}
        printf("Updating Q took ");
        printf("%.2lfs.\n", omp_get_wtime() - starttime);
        //printf("%.2lfs.\n", (double)(clock() - starttime) / CLOCKS_PER_SEC);
        fflush(stdout);
    }

    double checkConvergence() {
        // Frobenius norm used
        double diff_P = 100000000, diff_Q = 100000000;
        if(assigned) {
            diff_P = (P - P_current).norm();
            diff_Q = (Q - Q_current).norm();
            printf("Difference - P : %lf, Q : %lf\n", diff_P, diff_Q);
        }
        P_current = P;
        Q_current = Q;
        assigned = true;
        return max(diff_P, diff_Q);
    }
};


int main() {
	omp_set_num_threads(4);
    int num_factor, max_epoch;
    double lambda, confidence, diff_threshold;
    bool verbose;
    ifstream config_dir("./cfg.json");
    CharReaderBuilder builder;
    Value hyperparameters;
    string errs, fname_train, fname_val;
    bool ok = parseFromStream(builder, config_dir, &hyperparameters, &errs);
    if(ok) {
        num_factor = hyperparameters["num_factor"].asInt();
        max_epoch = hyperparameters["max_epoch"].asInt();
        lambda = hyperparameters["lambda"].asDouble();
        confidence = hyperparameters["confidence"].asDouble();
        diff_threshold = hyperparameters["diff_threshold"].asDouble();
        verbose = hyperparameters["verbose"].asBool();
        fname_train = hyperparameters["fname_train"].asString();
        fname_val = hyperparameters["fname_val"].asString();
    }
    else {
        cout << "parsing json failed" << endl;
        exit(0);
    }
    printf("num_factor : %d\n", num_factor);
    printf("lambda : %lf\n", lambda);
    printf("confidence : %lf\n", confidence);
    eALS eals = eALS(num_factor, max_epoch, lambda, \
                     confidence, diff_threshold, verbose, \
                     fname_train, fname_val);
    eals.loadData();
    eals.train();
    ofstream outFile_u("user.txt", ios::trunc);
    outFile_u << fixed;
    outFile_u << setprecision(8);
    //for(int i=0; i<eals.num_user; i++) {
    for(int i=eals.num_user - eals.val_user; i<eals.num_user; i++) {
        vector<double> logit(num_factor);
        VectorXd::Map(&logit[0], eals.P.row(i).size()) = eals.P.row(i);
        for(double i : logit) {
            outFile_u << i << " ";
        }
        outFile_u << endl;
    }
    outFile_u.close();
    ofstream outFile_s("item.txt", ios::trunc);
    outFile_s << fixed;
    outFile_s << setprecision(8);
    for(int i=0; i<eals.num_item; i++) {
        vector<double> logit(num_factor);
        VectorXd::Map(&logit[0], eals.Q.row(i).size()) = eals.Q.row(i);
        for(double i : logit) {
            outFile_s << i << " ";
        }
        outFile_s << endl;
    }
    outFile_s.close();
    config_dir.close();
    return 0;
}

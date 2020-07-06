// elementwise ALS
#include <iostream>
#include <string>
#include <algorithm> 
#include <random>
#include <time.h>
#include <fstream>
#include "eigen3/Eigen/Dense"
#include "jsoncpp/json.h"


using namespace std;
using namespace Eigen;
using namespace Json;

typedef Matrix<double, Dynamic, Dynamic> MatrixXd;

auto rng = std::default_random_engine {};
default_random_engine generator(time(0));

double random(double left, double right) {
  uniform_real_distribution<double> distribution(left, right);
  return distribution(generator);
}

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

class MatrixCoo {
public:
  vector<Info> data;
  int row, col, N, prev_N;
  MatrixCoo() {
  	  row = col = N = prev_N = 0;
  }

  void read(string fname) {
  	int r, c, n;
    FILE *in = fopen(fname.c_str(), "r");
    fscanf(in,"%d%d%d",&r, &c, &n);
    row += r, prev_N = N, N += n;
    col = c;
    data.resize(N);
    for(int i=prev_N;i<N;i++) {
      fscanf(in,"%d%d%lf",&data[i].user, &data[i].item, &data[i].score);
      // data[i].user --;
      // data[i].item --;
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
  MatrixXd X, Y, X_current, Y_current;
  MatrixCoo coodata, valdata;
  vector<int> col_num, row_num, mask_validate_user;
  vector<vector<int>> ground_truth;
  vector<vector<Element>> rui_hat, riu_hat;

  void loadData() {
    coodata.read(fname_train);
    int num_tmp = coodata.row;
    coodata.read(fname_val);
    val_user = coodata.row - num_tmp;
    num_user = coodata.row;
    num_item = coodata.col;
    printf("Matrix density : %lf\n", (double)coodata.N / num_user / num_item);
    printf("N : %d\n", coodata.N);
    printf("# of users : %d, # of items : %d\n", num_user, num_item);
    fflush(stdout);
  }
  /*
  void train_val_split() {
    int valsize = coodata.N * validation;
    shuffle(coodata.data.begin(), coodata.data.end(), rng);
    traindata.row = valdata.row = coodata.row;
    traindata.col = valdata.col = coodata.col;
    traindata.data.assign(coodata.data.begin() + valsize, coodata.data.end());
    traindata.N = (int)traindata.data.size();
    valdata.data.assign(coodata.data.begin(), coodata.data.begin() + valsize);
    valdata.N = (int)valdata.data.size();
  }
  */
  void train_validate() {
    X = MatrixXd::Random(num_user, num_factor); 
    X_current.resize(num_user, num_factor); 
    Y = MatrixXd::Random(num_item, num_factor); 
    Y_current.resize(num_item, num_factor);
    getRhat();
    // getGT();
    for(int iter=0;iter<max_epoch;iter++) {
      printf("Epoch %3d start\n", iter+1);
      fflush(stdout);
      train();
      double diff = checkConvergence();
      if(verbose) {
        if(diff<diff_threshold) {
          printf("\nEpoch : %d\n", iter + 1);
          printf("X and Y converged.\n");
          break;
        }
        if(iter==max_epoch-1) {
          printf("\nEpoch : %d\n", iter + 1);
          printf("Reached maximum iteration.\n");
        }
      }
    }
    // validate();
  }
  void getRhat() {
    int r, c;
    rui_hat.assign(X.rows(), vector<Element>());
    col_num.assign(X.rows(), 0);
    riu_hat.assign(Y.rows(), vector<Element>());
    row_num.assign(Y.rows(), 0);
    for(int i=0;i<coodata.N;i++) {
      r = coodata.data[i].user;
      c = coodata.data[i].item;
      rui_hat[r].push_back({X.row(r).dot(Y.row(c)), c, NULL});
      riu_hat[c].push_back({rui_hat[r][col_num[r]].val, r, NULL});
      col_num[r]++;
      row_num[c]++;
    }
    col_num.assign(X.rows(), 0);
    row_num.assign(Y.rows(), 0);
    for(int i=0;i<coodata.N;i++) {
      r = coodata.data[i].user;
      c = coodata.data[i].item;
      rui_hat[r][col_num[r]].dual = &riu_hat[c][row_num[c]];
      riu_hat[c][row_num[c]].dual = &rui_hat[r][col_num[r]];
      col_num[r]++;
      row_num[c]++;
    }
  }
  void train() {
    clock_t starttime;
    double puf, qif, numerator, denominator;
    vector<double> rui_hat_f, riu_hat_f;

    // train user factors
    starttime = clock();
    MatrixXd Sq(num_factor, num_factor);
    Sq = Y.transpose() * Y;
    MatrixXd SqT = Sq.transpose();
    for(int u=0;u<X.rows();u++) {
      for(int f=0;f<num_factor;f++) {
        rui_hat_f.resize(col_num[u]);
        numerator = X(u, f) * Sq(f, f) - X.row(u).dot(SqT.row(f));
        denominator = Sq(f, f) + lambda;
        for(int i_ind=0;i_ind<col_num[u];i_ind++) {
          qif = Y(rui_hat[u][i_ind].ind, f);
          rui_hat_f[i_ind] = rui_hat[u][i_ind].val - X(u, f) * qif;
          numerator += (1 + confidence - confidence * rui_hat_f[i_ind]) * qif;
          denominator += confidence * qif * qif;
        }
        X(u, f) = numerator / denominator;
        for(int i_ind=0;i_ind<col_num[u];i_ind++) {
          qif = Y(rui_hat[u][i_ind].ind, f);
          rui_hat[u][i_ind].val = X(u, f) * qif + rui_hat_f[i_ind]; 
        }
      }
    }
    for(int u=0;u<X.rows();u++) 
	  for(int i_ind=0;i_ind<col_num[u];i_ind++) 
	    rui_hat[u][i_ind].dual->val = rui_hat[u][i_ind].val;
    printf("Updating X took ");
    printf("%.2lfs.\n", (double)(clock() - starttime) / CLOCKS_PER_SEC);
    fflush(stdout);

    // train item factors
    starttime = clock();
    MatrixXd Sp(num_factor, num_factor);
    Sp = X.transpose() * X;
    MatrixXd SpT = Sp.transpose();
    for(int i=0;i<Y.rows();i++) {
      for(int f=0;f<num_factor;f++) {
        riu_hat_f.resize(row_num[i]);
        numerator = Y(i, f) * Sp(f, f) - Y.row(i).dot(SpT.row(f));
        denominator = Sp(f, f) + lambda;
        for(int u_ind=0; u_ind<row_num[i]; u_ind++) {
          puf = X(riu_hat[i][u_ind].ind, f);
          riu_hat_f[u_ind] = riu_hat[i][u_ind].val - Y(i, f) * puf;
          numerator += (1 + confidence - confidence * riu_hat_f[u_ind]) * puf;
          denominator += confidence * puf * puf;
        }
        Y(i, f) = numerator/denominator;
        for(int u_ind=0;u_ind<row_num[i];u_ind++) {
          puf = X(riu_hat[i][u_ind].ind, f);
          riu_hat[i][u_ind].val = riu_hat_f[u_ind] + Y(i, f) * puf;
        }
      }
    }
    for(int i=0;i<Y.rows();i++) 
      for(int u_ind=0;u_ind<row_num[i];u_ind++) 
        riu_hat[i][u_ind].dual->val = riu_hat[i][u_ind].val;
    printf("Updating Y took ");
    printf("%.2lfs.\n", (double)(clock() - starttime) / CLOCKS_PER_SEC);
    fflush(stdout);
  }
  /*
  void getGT() {
  	vector<int>::iterator it;
    mask_validate_user.resize(0);
    for(int i=0;i<num_user;i++)
      if(random(0.0,1.0)<validate_user)
        mask_validate_user.push_back(i);
    val_user = (int)mask_validate_user.size();
    ground_truth.resize(val_user, vector<int>());
    for(Info i : valdata.data) {
      it = find(mask_validate_user.begin(), mask_validate_user.end(), i.user);
      if(it!=mask_validate_user.end())
        ground_truth[distance(mask_validate_user.begin(), it)].push_back(i.item);
    }
  }
  */
  double checkConvergence() {
    double diff_X = 1, diff_Y = 1;
    if(assigned) {
      diff_X = diff_Y = 0;
      for(int i=0;i<X.rows();i++)
        for(int j=0;j<X.cols();j++)
          diff_X += (X_current(i, j) - X(i, j)) \
          			* (X_current(i, j) - X(i, j));
      for(int i=0;i<Y.rows();i++)
        for(int j=0;j<Y.cols();j++)
          diff_Y += (Y_current(i, j) - Y(i, j)) \
          	  * (Y_current(i, j) - Y(i, j));
      diff_X = sqrt(diff_X / X.rows() / X.cols());
      diff_Y = sqrt(diff_Y / Y.rows() / Y.cols());
      printf("Difference - X : %lf, Y : %lf\n", diff_X, diff_Y);
    }
    X_current = X;
    Y_current = Y;
    assigned = true;
    return max(diff_X, diff_Y);
  }
  /*
  void validate() {
  	double NDCG;
  	vector<int>::iterator it;
    vector<int> argsort(num_item);
    vector<double> tmp(num_item);
    vector<vector<int>> predict_idx(val_user);
    vector<vector<double>> predict(val_user);
    for(int i=0;i<val_user;i++) {
      for(int j=0;j<num_item;j++)
        tmp[j] = dot(X.val[mask_validate_user[i]], Y.val[j]);
      predict[i] = tmp;
    }
    for(Info i : traindata.data) {
      it = find(mask_validate_user.begin(), mask_validate_user.end(), i.user);
      if(it!=mask_validate_user.end())
        predict[distance(mask_validate_user.begin(), it)][i.item]=0.0;
    }
    for(int i=0;i<val_user;i++) {
      for(int j=0;j<num_item;j++)
        argsort[j] = j;
      sort(argsort.begin(), argsort.end(), \
      	   [&](int a, int b) { return predict[i][a] > predict[i][b]; });
      predict_idx[i] = argsort;
    }
    NDCG = getNDCG(predict_idx, ground_truth);
    printf("NDCG : %lf\n", NDCG);
  }
  double getNDCG(const vector<vector<int>> &pr, const vector<vector<int>> &re) {
    double res = 0.0, DCG, IDCG;
    printf("Validate user size : %d\n", val_user);
    for(int i=0;i<val_user;i++) {
      DCG = IDCG = 0.0;
      vector<int> real = re[i];
      vector<int> pred;
      if(!real.size()) continue;
      pred.assign(pr[i].begin(), pr[i].begin() + topn);
      sort(real.begin(), real.end());
      for(int j=0;j<topn;j++)
        if(binary_search(real.begin(),real.end(),pred[j]))
          DCG += 1.0 / log(j + 2.0);
      for(int j=0;j<min(topn,(int)real.size());j++)
        IDCG += 1.0 / log(j + 2.0);
      res += DCG / IDCG;
    }
    return res / (double)val_user;
  }
  */
};


int main() {
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
    eALS eals = eALS(num_factor, max_epoch, lambda, \
    		         confidence, diff_threshold, verbose, \
    		         fname_train, fname_val);
    eals.loadData();
    // eals.train_val_split();
    eals.train_validate();
    ofstream outFile_u("user.txt", ios::trunc);
    outFile_u << fixed;
    outFile_u << setprecision(10);
    for(int i=eals.num_user - eals.val_user; i<eals.num_user; i++) {
    	vector<double> logit(num_factor);
		VectorXd::Map(&logit[0], eals.X.row(i).size()) = eals.X.row(i);
    	for(double i : logit) {
    		outFile_u << i << " ";
		}
		outFile_u << endl;
	}
	outFile_u.close();
    ofstream outFile_s("song.txt", ios::trunc);
    outFile_s << fixed;
    outFile_s << setprecision(10);
    for(int i=0; i<eals.num_item; i++) {
    	vector<double> logit(num_factor);
		VectorXd::Map(&logit[0], eals.Y.row(i).size()) = eals.Y.row(i);
    	for(double i : logit) {
    		outFile_s << i << " ";
		}
		outFile_s << endl;
	}
	outFile_s.close();
	config_dir.close();
    return 0;
}
 

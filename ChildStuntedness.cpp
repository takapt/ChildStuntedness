#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
#include <cfloat>
#include <ctime>
#include <cassert>
#include <map>
#include <utility>
#include <set>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <sstream>
#include <complex>
#include <stack>
#include <queue>
#include <numeric>
#include <list>
#include <iomanip>
#include <fstream>
#include <bitset>

using namespace std;

#define foreach(it, c) for (__typeof__((c).begin()) it=(c).begin(); it != (c).end(); ++it)
template <typename T> void print_container(ostream& os, const T& c) { const char* _s = " "; if (!c.empty()) { __typeof__(c.begin()) last = --c.end(); foreach (it, c) { os << *it; if (it != last) os << _s; } } }
template <typename T> ostream& operator<<(ostream& os, const vector<T>& c) { print_container(os, c); return os; }
template <typename T> ostream& operator<<(ostream& os, const set<T>& c) { print_container(os, c); return os; }
template <typename T> ostream& operator<<(ostream& os, const multiset<T>& c) { print_container(os, c); return os; }
template <typename T> ostream& operator<<(ostream& os, const deque<T>& c) { print_container(os, c); return os; }
template <typename T, typename U> ostream& operator<<(ostream& os, const map<T, U>& c) { print_container(os, c); return os; }
template <typename T, typename U> ostream& operator<<(ostream& os, const pair<T, U>& p) { os << "(" << p.first << ", " << p.second << ")"; return os; }

template <typename T> void print(T a, int n, const string& split = " ") { for (int i = 0; i < n; i++) { cout << a[i]; if (i + 1 != n) cout << split; } cout << endl; }
template <typename T> void print2d(T a, int w, int h, int width = -1, int br = 0) { for (int i = 0; i < h; ++i) { for (int j = 0; j < w; ++j) { if (width != -1) cout.width(width); cout << a[i][j] << ' '; } cout << endl; } while (br--) cout << endl; }
template <typename T> void input(T& a, int n) { for (int i = 0; i < n; ++i) cin >> a[i]; }
#define dump(v) (cerr << #v << ": " << v << endl)

#define rep(i, n) for (int i = 0; i < (int)(n); ++i)
#define erep(i, n) for (int i = 0; i <= (int)(n); ++i)
#define all(a) (a).begin(), (a).end()
#define rall(a) (a).rbegin(), (a).rend()
#define clr(a, x) memset(a, x, sizeof(a))
#define sz(a) ((int)(a).size())
#define mp(a, b) make_pair(a, b)
#define ten(n) ((long long)(1e##n))

template <typename T, typename U> void upmin(T& a, const U& b) { a = min<T>(a, b); }
template <typename T, typename U> void upmax(T& a, const U& b) { a = max<T>(a, b); }
template <typename T> void uniq(T& a) { sort(a.begin(), a.end()); a.erase(unique(a.begin(), a.end()), a.end()); }
template <class T> string to_s(const T& a) { ostringstream os; os << a; return os.str(); }
template <class T> T to_T(const string& s) { istringstream is(s); T res; is >> res; return res; }
void fast_io() { cin.tie(0); ios::sync_with_stdio(false); }
bool in_rect(int x, int y, int w, int h) { return 0 <= x && x < w && 0 <= y && y < h; }

typedef long long ll;
typedef pair<int, int> pint;


class Random
{
private:
    unsigned int  x, y, z, w;
public:
    Random(unsigned int x
             , unsigned int y
             , unsigned int z
             , unsigned int w)
        : x(x), y(y), z(z), w(w) { }
    Random() 
        : x(123456789), y(362436069), z(521288629), w(88675123) { }
    Random(unsigned int seed)
        : x(123456789), y(362436069), z(521288629), w(seed) { }

    unsigned int next()
    {
        unsigned int t = x ^ (x << 11);
        x = y;
        y = z;
        z = w;
        return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    }

    int next_int() { return next(); }

    // [0, upper)
    int next_int(int upper) { return next() % upper; }

    // [low, high]
    int next_int(int low, int high) { return next_int(high - low + 1) + low; }

    double next_double(double upper) { return upper * next() / UINT_MAX; }
    double next_double(double low, double high) { return next_double(high - low) + low; }

    template <typename T>
    int select(const vector<T>& ratio)
    {
        T sum = accumulate(ratio.begin(), ratio.end(), (T)0);
        T v = next_double(sum) + (T)1e-6;
        for (int i = 0; i < (int)ratio.size(); ++i)
        {
            v -= ratio[i];
            if (v <= 0)
                return i;
        }
        return 0;
    }

    vector<int> choose(int n, int k)
    {
        vector<int> a(n);
        rep(i, n)
            a[i] = i;
        rep(i, n)
            swap(a[i], a[next_int(n)]);
        a.erase(a.begin() + k, a.end());
        return a;
    }
};

Random dc_tree_rand;
typedef vector<double> Feature;
struct DecisionTreeConfig
{
    int bag_size;
    int max_level;
    int leaf_size;
    int split_dimension_samples;

    DecisionTreeConfig()
        :
            bag_size(-1),
            max_level(-1),
            leaf_size(-1),
            split_dimension_samples(-1)
    {
    }
};
class DecisionTree
{
public:
    DecisionTree(const vector<Feature>& features, const vector<double>& label, const DecisionTreeConfig& config)
    {
        assert(features.size() == label.size());

        const int FEATURE_SIZE = features[0].size();

        const int max_level = config.max_level;
        const int leaf_size = config.leaf_size;
        const int SPLIT_DIMENSION_SAMPLES = config.split_dimension_samples;
        const int bag_size = max<int>(1, config.bag_size);
        assert(0 < SPLIT_DIMENSION_SAMPLES && SPLIT_DIMENSION_SAMPLES <= FEATURE_SIZE);
        assert(0 < bag_size && bag_size <= (int)features.size());

        vector<int> bag = dc_tree_rand.choose(features.size(), bag_size);


        nodes.push_back(TreeNode(0));

        // TODO: vector<int>はやめる。inplaceな配列一個持って、(left, right)をstackに突っ込む
        stack<pair<int, vector<int>>> sta;
        sta.push(make_pair(0, bag));
        while (!sta.empty())
        {
            int cur_node = sta.top().first;
            const vector<int> cur_features = sta.top().second;
            assert(!cur_features.empty());
            sta.pop();

            if (nodes[cur_node].level >= max_level || (int)cur_features.size() <= leaf_size)
            {
                double ave = 0;
                for (int i : cur_features)
                    ave += label[i];
                ave /= cur_features.size();
                nodes[cur_node].value = ave;

                double var = 0;
                for (int i : cur_features)
                    var += (label[i] - ave) * (label[i] - ave);
                var /= cur_features.size();
//                 printf("%3d, %3d: %.8f\n", sz(cur_features), nodes[cur_node].level, sqrt(var));

                continue;
            }

            const vector<int> split_dimensions = dc_tree_rand.choose(FEATURE_SIZE, SPLIT_DIMENSION_SAMPLES);
            int best_split_dimension = -1;
            double best_split_x = -1;
            double min_impurity = 1e60;
            for (int split_dimension : split_dimensions)
            {
                vector<pair<double, int>> x;
                for (int i : cur_features)
                    x.push_back(make_pair(features[i][split_dimension], i));
                sort(all(x));

                const double eps = 1e-9;
                if (x.front().first + eps > x.back().first)
                    continue;


                double right_squared_sum = 0;
                double right_sum = 0;
                for (auto& it : x)
                {
                    double t = label[it.second];
                    right_squared_sum += t * t;
                    right_sum += t;
                }
                int left_n = 0, right_n = x.size();
                double left_squared_sum = 0;
                double left_sum = 0;

                const int split_size_constraint = 1 + cur_features.size() / 20;
                const double split_lower_x = x.front().first + eps;
                const double split_upper_x = x.back().first - eps;
                rep(i, sz(cur_features) - split_size_constraint)
                {
                    const double t = label[x[i].second];

                    ++left_n;
                    left_squared_sum += t * t;
                    left_sum += t;

                    --right_n;
                    right_squared_sum -= t * t;
                    right_sum -= t;

                    if (i >= split_size_constraint && split_lower_x < x[i].first && x[i].first < split_upper_x)
                    {
                        // var(X) = E(X^2) - E(X)^2
                        // impurity = left_var * (left_n / n) + right_n * (right_n / n)
                        // denominator = var * n
                        double left_ave = left_sum / left_n;
                        double left_var_denominator = left_squared_sum - left_ave * left_ave;
                        double right_ave = right_sum / right_n;
                        double right_var_denominator = right_squared_sum - right_ave * right_ave;
                        double impurity = left_var_denominator + right_var_denominator;
                        if (impurity < min_impurity)
                        {
                            min_impurity = impurity;
                            best_split_dimension = split_dimension;
                            best_split_x = (x[i].first + x[i + 1].first) / 2;
                        }
                    }
                }
            }
            if (best_split_dimension == -1)
            {
                double ave = 0;
                for (int i : cur_features)
                    ave += label[i];
                ave /= cur_features.size();
                nodes[cur_node].value = ave;

                continue;
            }

            nodes[cur_node].split_dimension = best_split_dimension;
            nodes[cur_node].value = best_split_x;

            vector<int> left_features, right_features;
            for (int i : cur_features)
            {
                if (features[i][best_split_dimension] < best_split_x)
                    left_features.push_back(i);
                else
                    right_features.push_back(i);
            }
            if (left_features.empty())
            {
                vector<double> x;
                for (int i : cur_features)
                    x.push_back(features[i][best_split_dimension]);
                sort(all(x));
                dump(x);
                dump(best_split_x);
            }
            assert(!left_features.empty());
            assert(!right_features.empty());

//             printf("%3d %3d\n", (int)left_features.size(), (int)right_features.size());

            nodes[cur_node].left = nodes.size();
            nodes.push_back(TreeNode(nodes[cur_node].level + 1));
            sta.push(make_pair(nodes[cur_node].left, left_features));

            nodes[cur_node].right = nodes.size();
            nodes.push_back(TreeNode(nodes[cur_node].level + 1));
            sta.push(make_pair(nodes[cur_node].right, right_features));
        }
    }

    double predict(const Feature& x)
    {
        int cur_node = 0;
        while (!nodes[cur_node].is_leaf())
        {
            auto& no = nodes[cur_node];
            cur_node = x[no.split_dimension] < no.value ? no.left : no.right;
        }
        return nodes[cur_node].value;
    }

private:

    struct TreeNode
    {
        int level;
        int split_dimension;
        double value;

        int left;
        int right;

        TreeNode(int level)
            : level(level), split_dimension(-114514), left(-1919810), right(-364364)
        {
        }

        bool is_leaf() const
        {
            return split_dimension < 0;
        }
    };

    vector<TreeNode> nodes;
};
class RandomForest
{
public:
    RandomForest(const vector<Feature>& features, const vector<double>& labels)
    {
        const int FEATURE_SIZE = features[0].size();

        DecisionTreeConfig config;
        config.bag_size = features.size() / 2;
        config.max_level = 30;
        config.leaf_size = 50;
        config.split_dimension_samples = FEATURE_SIZE / 2;

        const int num_trees = 100;
        rep(tree_i, num_trees)
            trees.push_back(DecisionTree(features, labels, config));
    }

    double predict(const Feature& x)
    {
        double ave = 0;
        for (auto& tree : trees)
            ave += tree.predict(x);
        ave /= trees.size();
        return ave;
    }

private:
    vector<DecisionTree> trees;
};



const int ULTRA_SIZE = 8;
struct FetusRow
{
    double time;
    int sex; // 0 = Male, 1 = Female
    int status; // 1 or 2
    double ultra[ULTRA_SIZE];
};
struct Fetus
{
    int id;
    double weight;
    double duration;
    vector<FetusRow> rows;

    bool is_train_data() const
    {
        return weight >= 0;
    }
};

Feature create_feature(const Fetus& fetus)
{
    Feature feature;

    {
        const int SEGS = 23;
        rep(i, SEGS * ULTRA_SIZE)
            feature.push_back(0);
        for (auto& row : fetus.rows)
        {
            int seg = min<int>(SEGS - 1, row.time * SEGS);
            rep(i, ULTRA_SIZE)
            {
                feature[seg * ULTRA_SIZE + i] = row.ultra[i];
                if (seg > 0 && feature[(seg - 1) * ULTRA_SIZE + i] == 0)
                    feature[(seg - 1) * ULTRA_SIZE + i] = row.ultra[i];
                if (seg < ULTRA_SIZE - 1 && feature[(seg + 1) * ULTRA_SIZE + i] == 0)
                    feature[(seg + 1) * ULTRA_SIZE + i] = row.ultra[i];
            }
        }
    }

    {
        double min_t = 1e6, max_t = -1e6;
        for (auto& row : fetus.rows)
        {
            upmin(min_t, row.time);
            upmax(max_t, row.time);
        }
        feature.push_back(min_t);
        feature.push_back(max_t);
    }

    feature.push_back(fetus.rows.size());
    feature.push_back(fetus.rows.back().status);
    feature.push_back(fetus.rows.back().sex);

    return feature;
}

vector<Fetus> parse_train(const vector<string>& csv)
{
    map<int, Fetus> data;
    for (auto line : csv)
    {
        for (char& c : line)
            if (c == ',')
                c = ' ';

        FetusRow row;
        stringstream ss(line);
        int id;
        double weight, duration;
        ss >> id;
        ss >> row.time;
        ss >> row.sex;
        ss >> row.status;
        rep(i, ULTRA_SIZE)
            ss >> row.ultra[i];
        ss >> weight;
        ss >> duration;

        Fetus& fetus = data[id];
        fetus.id = id;
        fetus.weight = weight;
        fetus.duration = duration;
        data[id].rows.push_back(row);
    }

    vector<Fetus> fetus_data;
    for (auto& f : data)
        fetus_data.push_back(f.second);
    return fetus_data;
}
vector<Fetus> parse_test(const vector<string>& csv)
{
    map<int, Fetus> data;
    for (auto line : csv)
    {
        for (char& c : line)
            if (c == ',')
                c = ' ';

        FetusRow row;
        stringstream ss(line);
        int id;
        ss >> id;
        ss >> row.time;
        ss >> row.sex;
        ss >> row.status;
        rep(i, 8)
            ss >> row.ultra[i];

        Fetus& fetus = data[id];
        fetus.id = id;
        fetus.weight = fetus.duration = -1;
        data[id].rows.push_back(row);
    }

    vector<Fetus> fetus_data;
    for (auto& f : data)
        fetus_data.push_back(f.second);
    return fetus_data;
}

double calc_error(double weight, double duration, double correct_weight, double correct_duration)
{
    assert(0 <= correct_weight && correct_weight <= 1);
    assert(0 <= correct_duration && correct_duration <= 1);
    assert(0 <= weight && weight <= 1);
    assert(0 <= duration && duration <= 1);

    const static double inverseS[2][2] = {
         3554.42,  -328.119,
         -328.119,  133.511
    };

    double dw = weight - correct_weight;
    double dd = duration - correct_duration;

    double x = dd * inverseS[0][0] + dw * inverseS[0][1];
    double y = dd * inverseS[1][0] + dw * inverseS[1][1];

    double e = x * dd + y * dw;
    return e;
}

double calc_sse0(const vector<Fetus>& fetus_data)
{
    double ave_weight = 0;
    double ave_duration = 0;
    for (auto& fetus : fetus_data)
    {
        ave_weight += fetus.weight;
        ave_duration += fetus.duration;
    }
    ave_weight /= fetus_data.size();
    ave_duration /= fetus_data.size();

    double sse = 0;
    for (auto& fetus : fetus_data)
        sse += calc_error(fetus.weight, fetus.duration, ave_weight, ave_duration);
    return sse;
}

vector<string> read_file(const string& filename)
{
    fstream fs(filename);
    vector<string> lines;
    for (string s; fs >> s; )
        lines.push_back(s);
    return lines;
}


struct WeightDuration
{
    int id;
    double weight;
    double duration;
    WeightDuration()
        : id(-1), weight(-1), duration(-1)
    {
    }

    WeightDuration(int id, double weight, double duration)
        : id(id), weight(weight), duration(duration)
    {
    }

    WeightDuration(const Fetus& fetus)
        : id(fetus.id), weight(fetus.weight), duration(fetus.duration)
    {
        assert(0 <= id);
        assert(0 <= weight && weight <= 1);
        assert(0 <= duration && duration <= 1);
    }
};

double predict_by_dc(const vector<Fetus>& train_data, const vector<Fetus>& test_data)
{
    vector<Feature> train_features;
    vector<double> weight_label, duration_label;
    for (auto& train : train_data)
    {
        train_features.push_back(create_feature(train));
        weight_label.push_back(train.weight);
        duration_label.push_back(train.duration);

    }

    DecisionTreeConfig config;
    config.bag_size = train_data.size();
    config.max_level = 45;
    config.leaf_size = 50;
    config.split_dimension_samples = create_feature(train_data[0]).size();
    map<int, double> predict_weight;
    {
        DecisionTree tree(train_features, weight_label, config);
        for (auto& test : test_data)
            predict_weight[test.id] = tree.predict(create_feature(test));
    }

    map<int, double> predict_duration;
    {
        DecisionTree tree(train_features, duration_label, config);
        for (auto& test : test_data)
            predict_duration[test.id] = tree.predict(create_feature(test));
    }

    const double sse0 = calc_sse0(test_data);
    double sse = 0;
    for (auto& test : test_data)
        sse += calc_error(predict_weight[test.id], predict_duration[test.id], test.weight, test.duration);
    const double score = 1e6 * max(0.0, 1 - sse / sse0);

//     dump(sse0);
//     dump(sse);
//     dump(score);

    return score;
}

pair<map<int, double>, map<int, double>> predict(const vector<Fetus>& train_data, const vector<Fetus>& test_data)
{
    vector<Feature> train_features;
    vector<double> weight_label, duration_label;
    for (auto& train : train_data)
    {
        train_features.push_back(create_feature(train));
        weight_label.push_back(train.weight);
        duration_label.push_back(train.duration);
    }

    map<int, double> predict_weight;
    {
        RandomForest rf(train_features, weight_label);
        for (auto& test : test_data)
            predict_weight[test.id] = rf.predict(create_feature(test));
    }

    map<int, double> predict_duration;
    {
        RandomForest rf(train_features, duration_label);
        for (auto& test : test_data)
            predict_duration[test.id] = rf.predict(create_feature(test));
    }

    return make_pair(predict_weight, predict_duration);
}

double calc_score(const vector<Fetus>& test_data, map<int, double>& predict_weight, map<int, double>& predict_duration)
{
    const double sse0 = calc_sse0(test_data);
    double sse = 0;
    for (auto& test : test_data)
        sse += calc_error(predict_weight[test.id], predict_duration[test.id], test.weight, test.duration);
    double score = 1e6 * max(0.0, 1 - sse / sse0);
    return score;
}

class ChildStuntedness
{
public:
    vector<double> predict(vector<string>& training, vector<string>& testing)
    {
        auto train_data = parse_train(training);
        auto test_data = parse_test(testing);

        auto prepre = ::predict(train_data, test_data);
        auto& predict_weight = prepre.first;
        auto& predict_duration = prepre.second;

        vector<int> ids;
        for (auto& test : test_data)
            ids.push_back(test.id);
        sort(all(ids));
        vector<double> res;
        for (int id : ids)
        {
            assert(predict_weight.count(id));
            assert(predict_duration.count(id));
            res.push_back(predict_duration[id]);
            res.push_back(predict_weight[id]);
        }
        return res;
    }
};

#ifdef LOCAL
int main()
{
    const string train_filename = "exampleData.csv";

    vector<string> train_csv = read_file(train_filename);
    vector<Fetus> fetus_data = parse_train(train_csv);

    const int T = 200;
    vector<vector<Fetus>> train_data_set;
    vector<vector<Fetus>> test_data_set;
    rep(_, T)
    {
        const int train_size = fetus_data.size() * 0.66;
        train_data_set.push_back(vector<Fetus>(fetus_data.begin(), fetus_data.begin() + train_size));
        test_data_set.push_back(vector<Fetus>(fetus_data.begin() + train_size, fetus_data.end()));

        random_shuffle(all(fetus_data));
    }

    vector<double> scores(T);
#pragma omp parallel for
    rep(test_i, T)
    {
        const auto& train_data = train_data_set[test_i];
        const auto& test_data = test_data_set[test_i];

        auto res = predict(train_data, test_data);
        auto& predict_weight = res.first;
        auto& predict_duration = res.second;

        double score = calc_score(test_data, predict_weight, predict_duration);
        dump(score);

        scores[test_i] = score;
    }
    const double ave = accumulate(all(scores), 0.0) / T;
    dump(ave);

//     sort(all(scores));
//     dump(scores);
}
#endif

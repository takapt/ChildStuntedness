#define NDEBUG

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
#define dump(v) (cout << #v << ": " << v << endl)

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





#ifdef _MSC_VER
#include <Windows.h>
#else
#include <sys/time.h>
#endif
class Timer
{
    typedef double time_type;
    typedef unsigned int skip_type;

private:
    time_type start_time;
    time_type elapsed;

#ifdef _MSC_VER
    time_type get_ms() { return (time_type)GetTickCount64() / 1000; }
#else
    time_type get_ms() { struct timeval t; gettimeofday(&t, NULL); return (time_type)t.tv_sec * 1000 + (time_type)t.tv_usec / 1000; }
#endif

public:
    Timer() {}

    void start() { start_time = get_ms(); }
    time_type get_elapsed() { return elapsed = get_ms() - start_time; }
};

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
            swap(a[i], a[next_int(i + 1)]);
        a.erase(a.begin() + k, a.end());
        return a;
    }
};
Timer g_timer;


Random dc_tree_rand;
typedef vector<double> Feature;
struct DecisionTreeConfig
{
    int bag_size;
    int max_level;
    int min_split_size;
    int split_dimension_samples;

    DecisionTreeConfig()
        :
            bag_size(-1),
            max_level(-1),
            min_split_size(-1),
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
        const int min_split_size = config.min_split_size;
        const int SPLIT_DIMENSION_SAMPLES = config.split_dimension_samples;
        const int bag_size = max<int>(1, config.bag_size);
        assert(0 < SPLIT_DIMENSION_SAMPLES && SPLIT_DIMENSION_SAMPLES <= FEATURE_SIZE);
        assert(0 < bag_size && bag_size <= (int)features.size());


        // TODO: 重複には(* count)の重みをかける
        vector<int> bag(bag_size);
        rep(i, bag_size)
            bag[i] = dc_tree_rand.next_int(features.size());


        nodes.push_back(TreeNode(0));

        stack<pair<int, vector<int>>> sta;
        sta.push(make_pair(0, bag));
        while (!sta.empty())
        {
            int cur_node = sta.top().first;
            const vector<int> cur_features = sta.top().second;
            assert(!cur_features.empty());
            sta.pop();

            double ave = 0;
            for (int i : cur_features)
                ave += label[i];
            ave /= cur_features.size();

            if (nodes[cur_node].level >= max_level || (int)cur_features.size() <= min_split_size)
            {
                double ave = 0;
                for (int i : cur_features)
                    ave += label[i];
                ave /= cur_features.size();
                nodes[cur_node].value = ave;

                continue;
            }

            int best_split_dimension = -1;
            double best_split_x = -1;
            double min_impurity = 1e60;
            const vector<int> split_dimensions = dc_tree_rand.choose(FEATURE_SIZE, SPLIT_DIMENSION_SAMPLES);
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

                const int split_size_constraint = max(1, min_split_size / 2);
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
                        double left_ave = left_sum / left_n;
                        double left_var = left_squared_sum / left_n - left_ave * left_ave;
                        double right_ave = right_sum / right_n;
                        double right_var = right_squared_sum / right_n - right_ave * right_ave;
                        double impurity = left_var * left_n + right_var * right_n;
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
            assert(!left_features.empty());
            assert(!right_features.empty());

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
    RandomForest(const vector<Feature>& features, const vector<double>& labels, const DecisionTreeConfig& config, int num_trees, double tle)
    {
        Timer timer;
        timer.start();
        while ((int)trees.size() < num_trees && timer.get_elapsed() < tle)
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

    bool operator<(const FetusRow& r) const
    {
        return time < r.time;
    }
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
        auto last = *max_element(all(fetus.rows));

        rep(i, ULTRA_SIZE)
            feature.push_back(last.ultra[i]);

        double ave = 0;
        rep(i, ULTRA_SIZE)
            ave += last.ultra[i];
        ave /= ULTRA_SIZE;

        double var = 0;
        rep(i, ULTRA_SIZE)
            var += (last.ultra[i] - ave) * (last.ultra[i] - ave);
        var /= ULTRA_SIZE;

        feature.push_back(ave);
        feature.push_back(var);


        auto first = *min_element(all(fetus.rows));

        feature.push_back(first.ultra[0]);

        feature.push_back(first.time);
        feature.push_back(last.time);

        feature.push_back(last.time - first.time);
    }

    feature.push_back(fetus.rows.size());
    feature.push_back(fetus.rows.back().status);
    feature.push_back(fetus.rows.back().sex);

    return feature;
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


#ifdef LOCAL
const double G_TLE = 1919810.0 * 1000;
#else
const double G_TLE = 120 * 1000;
#endif


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

    const int FEATURE_SIZE = train_features[0].size();
    const double RF_TLE = G_TLE * 0.43;
    const int MAX_TREES = 1200;

    map<int, double> predict_weight;
    {
        DecisionTreeConfig config;
        config.bag_size = train_features.size();
        config.max_level = 256;
        config.min_split_size = 40;
        config.split_dimension_samples = FEATURE_SIZE;
        RandomForest rf(train_features, weight_label, config, MAX_TREES, RF_TLE);
        for (auto& test : test_data)
            predict_weight[test.id] = rf.predict(create_feature(test));
    }

    map<int, double> predict_duration;
    {
        DecisionTreeConfig config;
        config.bag_size = train_features.size();
        config.max_level = 256;
        config.min_split_size = 80;
        config.split_dimension_samples = FEATURE_SIZE;
        RandomForest rf(train_features, duration_label, config, MAX_TREES, RF_TLE);
        for (auto& test : test_data)
            predict_duration[test.id] = rf.predict(create_feature(test));
    }

    return make_pair(predict_weight, predict_duration);
}

double calc_sse(const vector<Fetus>& test_data, map<int, double>& predict_weight, map<int, double>& predict_duration)
{
    double sse = 0;
    for (auto& test : test_data)
        sse += calc_error(predict_weight[test.id], predict_duration[test.id], test.weight, test.duration);
    return sse;
}
double calc_score(const vector<Fetus>& test_data, map<int, double>& predict_weight, map<int, double>& predict_duration)
{
    double sse0 = calc_sse0(test_data);
    double sse = calc_sse(test_data, predict_weight, predict_duration);
    double score = 1e6 * max(0.0, 1 - sse / sse0);
    return score;
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


#ifdef LOCAL
vector<string> read_file(const string& filename)
{
    fstream fs(filename);
    vector<string> lines;
    for (string s; fs >> s; )
        lines.push_back(s);
    return lines;
}
void write_features_labels(const string& filename, vector<Fetus> fetus_data)
{
    random_shuffle(all(fetus_data));
    ofstream fs(filename);
    for (auto& fetus : fetus_data)
    {
        auto feature = create_feature(fetus);

        for (auto x : feature)
            fs << x << " ";
        fs << fetus.weight << " " << fetus.duration << endl;
    }
    fs.close();
}

struct Result
{
    int id;
    double w, d, cw, cd;
    double last_t;
    Result(int id, double w, double d, double cw, double cd, double last_t)
        : id(id), w(w), d(d), cw(cw), cd(cd), last_t(last_t)
    {
    }

    double error() const
    {
        return calc_error(w, d, cw, cd);
    }

    string to_s(double sse0) const
    {
        char buf[128];
        sprintf(buf, "%5d, %.4f: %5.0f, (%+.4f, %+.4f), (%.4f, %.4f), (%.4f, %.4f)", id, last_t, 1e6 * error() / sse0, w - cw, d - cd, w, d, cw, cd);
        return buf;
    }

    string to_s_for_ana() const
    {
        stringstream ss;
        ss
            << id << " "        // 0
            << last_t << " "    // 1
            << w - cw << " "    // 2
            << d - cd << " "    // 3
            << w << " "         // 4
            << d << " "         // 5
            << cw << " "        // 6
            << cd;              // 7
        return ss.str();
    }

    bool operator<(const Result& r) const
    {
        return error() < r.error();
    }
};
vector<Result> get_results(const vector<Fetus>& test_data, map<int, double>& predict_weight, map<int, double>& predict_duration)
{
    vector<Result> results;
    for (auto& test : test_data)
        results.push_back(Result(test.id, predict_weight[test.id], predict_duration[test.id], test.weight, test.duration, max_element(all(test.rows))->time));
    return results;
}

int main()
{
    const string train_filename = "exampleData.csv";

    vector<string> train_csv = read_file(train_filename);

    g_timer.start();

    vector<Fetus> fetus_data = parse_train(train_csv);

//     write_features_labels("features.txt", fetus_data);
//     return 0;

    srand(time(NULL));
    const int T = 100;
    vector<vector<Fetus>> train_data_set;
    vector<vector<Fetus>> test_data_set;
    rep(_, T)
    {
        const int train_size = fetus_data.size() * 0.66;
        train_data_set.push_back(vector<Fetus>(fetus_data.begin(), fetus_data.begin() + train_size));
        test_data_set.push_back(vector<Fetus>(fetus_data.begin() + train_size, fetus_data.end()));

        random_shuffle(all(fetus_data));
    }

//     predict_by_dctree(train_data_set[0], test_data_set[0]);
//     return 0;

    vector<double> scores(T);
#pragma omp parallel for
    rep(test_i, T)
    {
        const auto& train_data = train_data_set[test_i];
        const auto& test_data = test_data_set[test_i];

        auto res = predict(train_data, test_data);
//         auto res = predict_by_dctree(train_data, test_data);
        auto& predict_weight = res.first;
        auto& predict_duration = res.second;

        double score = calc_score(test_data, predict_weight, predict_duration);

        double sse0 = calc_sse0(test_data);
        double sse = calc_sse(test_data, predict_weight, predict_duration);
        if (score < 380000 && false)
        {
            dump(score);
            dump(sse0);
            dump(sse);

            auto results = get_results(test_data, predict_weight, predict_duration);
            sort(all(results));
//             rep(i, 10)
//                 cout << results[i].to_s(sse0) << endl;
//             cout << endl;
//             rep(i, results.size())
//                 cout << results[results.size() - 1 - i].to_s_for_ana() << endl;
            rep(i, 40)
                cout << results[results.size() - 1 - i].to_s(sse0) << endl;
        }

        scores[test_i] = score;
    }
    const double ave = accumulate(all(scores), 0.0) / T;
    dump(ave);

//     sort(all(scores));
//     dump(scores);
}
#endif


class ChildStuntedness
{
public:
    vector<double> predict(vector<string>& training, vector<string>& testing)
    {
        g_timer.start();

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

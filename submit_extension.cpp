#ifndef LOCAL
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
#endif

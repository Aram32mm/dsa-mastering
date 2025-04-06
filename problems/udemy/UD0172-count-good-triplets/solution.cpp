#include <bits/stdc++.h>
using namespace std;

#define FAST_IO ios::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr);
#define ll long long
#define vi vector<int>
#define vvi vector<vector<int>>
#define pii pair<int, int>
#define pb push_back
#define all(x) x.begin(), x.end()


#ifndef ONLINE_JUDGE
void setIO() {
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
}
#else
void setIO() {}
#endif

// O(n^3) time | O(1) space
vvi findTripletsWithGPBF(const vi& nums, int r){
    int n =  nums.size();
    vvi res;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j){
            if (nums[j] != nums[i] * r) continue;
            for (int k = j + 1; k < n; ++k) {
                if (nums[k] == nums[j] * r) {
                    res.pb({nums[i], nums[j], nums[k]});
                }
            }
        }
    }
    return res;
}


vvi findTripletsWithGP(const vi& nums, int r) {
    unordered_map<ll, ll> left, right;
    for (int num : nums) right[num]++;

    vvi res;

    for (int num : nums) {

        right[num]--;

        if (num % r == 0) {

            ll a = left[num / r];
            ll c = right[num * r];

            for (ll i = 0; i < a * c; ++i) {
                res.pb({(int)(num / r), num, (int)(num * r)});
            }

        }

        left[num]++;

    }

    return res;
}

void solve() {
    int n, r;
    cin >> n >> r;
    vi nums(n);
    for (int& x : nums) cin >> x;

    vvi result = findTripletsWithGP(nums, r);
    for (const auto& triplet : result) {
        for (int x : triplet) cout << x << " ";
        cout << "\n";
    }
    cout << "---\n";
}

int main() {
    FAST_IO;
    setIO();

    int T;
    cin >> T;
    while (T--) {
        solve();
    }

    return 0;
}

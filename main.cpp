#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "opencv2/opencv.hpp"

//https://oi-wiki.org/math/numerical/interp/
//https://www.luogu.com.cn/problem/solution/CF622F

const int eps = 1e-6;

int n, m;
double lambda;
std::vector<cv::Point2f>control_points;

struct Set_Color {
    int r, g, b;
    Set_Color(int rr = 0, int gg = 0, int bb = 0) {
        r = rr; g = gg; b = bb;
    }
    Set_Color() {
        r = g = b = 0;
    }
};

void mouse_handler(int event, int x, int y, int flags, void* userdata) {

    if (event == cv::EVENT_LBUTTONDOWN && control_points.size() < n) {

        std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << '\n';
        control_points.emplace_back(x, y);
    }
}

inline bool check(cv::Point2f p) {
    return p.x > 0 && p.x < 700 && p.y > 0 && p.y < 700;
}

void antialiasingShading(const cv::Point2f& lastpoint, const cv::Point2f& point, Set_Color color, cv::Mat& window) {

    float d = 2, extendfactor = 0;

    cv::Vec2f tanVec = cv::Vec2f{ point.x - lastpoint.x, point.y - lastpoint.y };
    tanVec = tanVec / sqrt(tanVec[0] * tanVec[0] + tanVec[1] * tanVec[1]);

    cv::Vec2f normalVec = cv::Vec2f{ -tanVec[1], tanVec[0] } *d;

    cv::Point2f q1 = cv::Point2f{ lastpoint.x + normalVec[0] - tanVec[0] * extendfactor,
                                lastpoint.y + normalVec[1] - tanVec[1] * extendfactor };
    cv::Point2f q2 = cv::Point2f{ lastpoint.x - normalVec[0] - tanVec[0] * extendfactor,
                                lastpoint.y - normalVec[1] - tanVec[1] * extendfactor };
    cv::Point2f q3 = cv::Point2f{ point.x + normalVec[0] + tanVec[0] * extendfactor,
                                point.y + normalVec[1] + tanVec[1] * extendfactor };
    cv::Point2f q4 = cv::Point2f{ point.x - normalVec[0] + tanVec[0] * extendfactor,
                                point.y - normalVec[1] + tanVec[1] * extendfactor };

    int minx = std::min(q1.x, std::min(q2.x, std::min(q3.x, q4.x)));
    int maxx = std::max(q1.x, std::max(q2.x, std::max(q3.x, q4.x)));
    int miny = std::min(q1.y, std::min(q2.y, std::min(q3.y, q4.y)));
    int maxy = std::max(q1.y, std::max(q2.y, std::max(q3.y, q4.y)));

    for (int i = minx; i <= maxx; i++)
        for (int j = miny; j <= maxy; j++) {

            cv::Vec2f newVec = cv::Vec2f{ (float)i + 0.5f - lastpoint.x, (float)j + 0.5f - lastpoint.y };
            float len = sqrt(newVec[0] * newVec[0] + newVec[1] * newVec[1]);
            newVec = newVec / len;

            float dis = std::min((float)sqrt(1 - pow(tanVec.dot(newVec), 2)) * len, d);

            cv::Point2f p1 = cv::Point2f{ (float)i + 0.25f, (float)j + 0.25f };
            cv::Point2f p2 = cv::Point2f{ (float)i + 0.75f, (float)j + 0.25f };
            cv::Point2f p3 = cv::Point2f{ (float)i + 0.25f, (float)j + 0.75f };
            cv::Point2f p4 = cv::Point2f{ (float)i + 0.75f, (float)j + 0.75f };

            int param = 0;

            newVec = cv::Vec2f{ p1.x - lastpoint.x, p1.y - lastpoint.y };
            len = sqrt(newVec[0] * newVec[0] + newVec[1] * newVec[1]);
            newVec = newVec / len;
            param += sqrt(1 - pow(tanVec.dot(newVec), 2)) * len <= d;

            newVec = cv::Vec2f{ p2.x - lastpoint.x, p2.y - lastpoint.y };
            len = sqrt(newVec[0] * newVec[0] + newVec[1] * newVec[1]);
            newVec = newVec / len;
            param += sqrt(1 - pow(tanVec.dot(newVec), 2)) * len <= d;

            newVec = cv::Vec2f{ p3.x - lastpoint.x, p3.y - lastpoint.y };
            len = sqrt(newVec[0] * newVec[0] + newVec[1] * newVec[1]);
            newVec = newVec / len;
            param += sqrt(1 - pow(tanVec.dot(newVec), 2)) * len <= d;

            newVec = cv::Vec2f{ p4.x - lastpoint.x, p4.y - lastpoint.y };
            len = sqrt(newVec[0] * newVec[0] + newVec[1] * newVec[1]);
            newVec = newVec / len;
            param += sqrt(1 - pow(tanVec.dot(newVec), 2)) * len <= d;

            if (color.r)
                window.at<cv::Vec3b>(j, i)[0] = std::max(window.at<cv::Vec3b>(j, i)[1], (uchar)(255.0 * (1 - dis / d) * (float)param / 4.0));
            if (color.g)
                window.at<cv::Vec3b>(j, i)[1] = std::max(window.at<cv::Vec3b>(j, i)[1], (uchar)(255.0 * (1 - dis / d) * (float)param / 4.0));
            if (color.b)
                window.at<cv::Vec3b>(j, i)[2] = std::max(window.at<cv::Vec3b>(j, i)[1], (uchar)(255.0 * (1 - dis / d) * (float)param / 4.0));
        }
}


cv::Point2f Lagrange_function(float t) {
    float y = 0.0f, temp = 1.0f;

    for (int i = 0; i < n; i++) {
        temp = 1.0f;
        for (int j = 0; j < n; j++) {
            if (i == j)continue;
            temp *= (t - control_points[j].x) / (control_points[i].x - control_points[j].x);
        }
        y += temp * control_points[i].y;
    }

    return cv::Point2f(t, y);
}
// Lagrange 插值函数
void Lagrange(const std::vector<cv::Point2f>& control_points, cv::Mat& window) {

    cv::Point2f lastpoint = control_points[0];
    
    for (float t = control_points[0].x + 1.0; t <= control_points[n - 1].x; t += 1.0) {
        auto point = Lagrange_function(t);
        if (check(lastpoint) && check(point))
            antialiasingShading(lastpoint, point, Set_Color(1, 0, 0), window);
        lastpoint = point;
    }
}


double sigma = 1.0;
struct Gauss {
    double a[5], b;
}g[5];
double b0;
double b[5];
cv::Point2f Gauss_function(float t) {
    float y = b0;
    for (int i = 0; i < n; i++) {
        y += b[i] * exp(-(t - control_points[i].x) * (t - control_points[i].x) / (2.0 * sigma));
    }
    return cv::Point2f(t, y);
}
int Gauss_Elimination(std::vector<int>& free_elem) {

    int nxtval = -1;
    for (int k = 0; k < n; k++) {

        nxtval++;
        while (nxtval < n) {
            for (int i = k + 1; i < n; i++)
                if (fabs(g[i].a[nxtval]) > fabs(g[k].a[nxtval]))std::swap(g[i], g[k]);
            if (fabs(g[k].a[nxtval]) < eps) {
                free_elem.push_back(nxtval);
                nxtval++;
            }
            else break;
        }
        if (nxtval == n) {
            for (int i = k; i < n; i++) {
                // No Solution
                if (fabs(g[i].b) > eps)return -1;
            }
            // Multiple Solution
            return 1;
        }

        double s = g[k].a[nxtval];
        for (int i = nxtval; i < n; i++)g[k].a[i] /= s;
        g[k].b /= s;
        for (int i = 0; i < n; i++) {
            if (i == k)continue;
            s = g[i].a[nxtval];
            for (int j = nxtval; j < n; j++)
                g[i].a[j] -= (g[k].a[j] * s);
            g[i].b -= (g[k].b * s);
        }
    }
    return 0;
}
int Gauss_init(const std::vector<cv::Point2f>& control_points) {

    b0 = 0;
    for (int i = 0; i < n; i++) {
        g[i].b = control_points[i].y;
        b0 += g[i].b;
    }
    b0 /= n;
    for (int i = 0; i < n; i++)
        g[i].b -= b0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            g[i].a[j] = std::exp(-1.0 * (control_points[i].x - control_points[j].x) * (control_points[i].x - control_points[j].x) / (2.0 * sigma));
    for (int i = 0; i < n; i++)g[i].a[i] = 1;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            g[i].a[j] = g[j].a[i];

    std::vector<int>free_elem;
    free_elem.clear();
    if (Gauss_Elimination(free_elem) == -1) {
        return -1;
    }
    for (int i = 0; i < (int)free_elem.size(); i++) {
        b[free_elem[i]] = b0;
    }
    for (int i = n - 1; i >= 0; i--) {
        int id = i;
        for (; id < n; id++) {
            if (fabs(g[i].a[id]) > eps)break;
        }
        if (id == n)continue;
        b[id] = g[i].b;
        for (int j = id + 1; j < n; j++) {
            if (fabs(g[i].a[j]) > eps) {
                b[id] -= g[i].a[j] * b[j];
            }
        }
    }
    return 0;
}
// Gauss 插值函数
void Gauss(const std::vector<cv::Point2f>& control_points, cv::Mat& window) {

    if (Gauss_init(control_points) == -1) {
        printf("Gauss Interpolation NO SOLUTION!\n");
        return;
    }

    cv::Point2f lastpoint = control_points[0];
    for (float t = control_points[0].x + 1.0; t <= control_points[n - 1].x; t += 1.0) {
        auto point = Gauss_function(t);
        if (check(lastpoint) && check(point))
            antialiasingShading(lastpoint, point, Set_Color(0, 1, 0), window);
        lastpoint = point;
    }

}


double raw_mat[5][5];
double mat[5][5];
cv::Point2f Lsm_function(float t) {
    float y = b[0];
    double temp = t;
    for (int i = 1; i < m; i++) {
        y += b[i] * temp;
        temp *= t;
    }
    return cv::Point2f(t, y);
}
int Lsm_init(const std::vector <cv::Point2f>& control_points, double lambda) {

    for (int i = 0; i < n; i++)raw_mat[i][0] = 1;
    for (int i = 0; i < n; i++) {
        double temp = control_points[i].x;
        for (int j = 1; j < m; j++) {
            raw_mat[i][j] = temp;
            temp *= (control_points[i].x);
        }
    }
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            mat[i][j] = 0.0;
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                mat[i][j] += raw_mat[k][i] * raw_mat[k][j];
            }
        }
    }
    for (int i = 0; i < m; i++)mat[i][i] += lambda;

    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            g[i].a[j] = mat[i][j];
    for (int i = 0; i < m; i++)g[i].b = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            g[j].b += raw_mat[i][j] * control_points[i].y;
    }

    std::vector<int>free_elem;
    free_elem.clear();
    if (Gauss_Elimination(free_elem) == -1) {
        return -1;
    }
    for (int i = 0; i < (int)free_elem.size(); i++) {
        b[free_elem[i]] = b0;
    }
    for (int i = n - 1; i >= 0; i--) {
        int id = i;
        for (; id < n; id++) {
            if (fabs(g[i].a[id]) > eps)break;
        }
        if (id == n)continue;
        b[id] = g[i].b;
        for (int j = id + 1; j < n; j++) {
            if (fabs(g[i].a[j]) > eps) {
                b[id] -= g[i].a[j] * b[j];
            }
        }
    }
    return 0;

}
// 最小二乘法回归
void Lsm(const std::vector <cv::Point2f>& control_points, double lambda, cv::Mat& window) {

    if (Lsm_init(control_points, lambda) == -1) {
        printf("Gauss Interpolation NO SOLUTION!\n");
        return;
    }

    cv::Point2f lastpoint = Lsm_function(control_points[0].x);
    for (float t = control_points[0].x + 1.0; t <= control_points[n - 1].x; t += 1.0) {
        auto point = Lsm_function(t);
        if (check(lastpoint) && check(point))
            antialiasingShading(lastpoint, point, Set_Color(0, 0, 1), window);
        lastpoint = point;
    }

}


int main() {

    printf("在键入点时，请从左至右逐一键入，上限为5个点\n");
    printf("请输入需要拟合的点的个数：\n");
    scanf("%d", &n);
    if (n > 5) {
        printf("点数超过5个，程序终止\n");
        return 0;
    }
    printf("请输入逼近型拟合函数的最高次数（不能超过n-1，与n-1取最小值）：\n");
    scanf("%d", &m);
    m = std::min(m, n - 1);
    m++;
    printf("请输入正则项权重（不能小于0）：\n");
    scanf("%lf\n", &lambda);

    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));
    cv::namedWindow("Function_fitting", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Function_fitting", mouse_handler, nullptr);

    int key = -1;
    while (key != 27) {

        for (auto& point : control_points) {

            cv::circle(window, point, 2, { 255, 255, 255 }, 3);
        }

        if (control_points.size() == n) {

            // 插值型拟合
            Lagrange(control_points, window);
            Gauss(control_points, window);

            // 逼近型拟合
            Lsm(control_points, lambda, window);

            cv::cvtColor(window, window, cv::COLOR_BGR2RGB);
            cv::imshow("Function_fitting", window);
            cv::imwrite("Function_fitting.png", window);
            key = cv::waitKey(0);

            return 0;
        }
        
        cv::imshow("Function_fitting", window);
        key = cv::waitKey(20);

    }


    return 0;
}
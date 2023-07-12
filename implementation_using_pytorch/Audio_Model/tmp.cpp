#include <iostream>

using namespace std;

int main()
{
    int arr[] = {5, 3, 2, 12, 40, 32, 8, 9, 56};
    int give;
    int ind_mapping[size(arr)], ans_arr[size(arr)];

    cout << "Enter an number: ";
    cin >> give;

    for (int i = 0; i < size(arr); i++)
    {
        ans_arr[i] = arr[i];

        // For tmp index
        ind_mapping[i] = i;
    }

    for (int i = 0; i < size(arr); i++)
    {
        ans_arr[i] = abs(ans_arr[i] - give);
    }

    for (int i = 0; i < size(arr); i++)
    {
        int min = i;
        for (int j = i; j < size(arr); j++)
        {
            // For sorting
            if (ans_arr[min] > ans_arr[j])
            {
                min = j;
            }
        }

        if (min != i)
        {
            // Also swap the mapping index
            swap(ind_mapping[i], ind_mapping[min]);

            swap(ans_arr[i], ans_arr[min]);
        }
    }

    for (int k = 0; k < 3; k++)
    {
        cout << arr[ind_mapping[k]] << " ";
    }
}
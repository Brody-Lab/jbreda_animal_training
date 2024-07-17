**Goal:** 

date as row index dataframe with summary information from water, sessions and mass table for all animals.

**Difficulties:**
1. keys are difference across tables (e.g. date, date_val)
2. if there are multiple sessions in a day for an animal, aggregations differ across values.
3.

**Current inefficiencies:**
1. In `create_days_df_from_dj` and associated sub functions, I load information a single day at a time, aggregate it and then append as a row to a df and repeat. This is really, really slow and doesn't utilize any of the SQL backend

2. If I run the code after the animals have been weighed, but before they finish training, the session data for that day is ignored. The code sees there is a row in the df with that date and moves on.


**Join Keys:**

ID: `Sessions.ratname`, `Mass.ratname`, `Water.rat`, `Rigwater.ratname`
Date: `Sesssions.sessiondate`, `Mass.date`, `Water.date`, `Rigwater.dateval`


**Items to fetch:**

| **table** |     **bdata key**    |              **agg**              |
|:---------:|:--------------------:|:---------------------------------:|
|  Sessions |    `n_done_trials`   |                sum                |
|  Sessions |      `hostname`      |              grab any             |
|  Sessions |      `starttime`     |             grab first            |
|  Sessions |       `endtime`      |             grab last             |
|  Sessions |    `total_correct`   | weighted avg given  n_done_trials |
|  Sessions | `percent_violations` | weighted avg given  n_done_trials |
|  Sessions |    `right_correct`   | weighted avg given  n_done_trials |
|  Sessions |    `left_correct`    | weighted avg given  n_done_trials |
|    Mass   |        `mass`        |         None, but 0 -> Nan        |
|   Water   |   `percent_target`   |                None               |
|   Water   |       `volume`       |    Take max  (sometimes has 0s)   |
|  Rigwater |      `totalvol`      |                None               |
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:49:49 2018

@author: Mathieu Daviet
"""

import get_data as gd
import manage_data as md
import cls

[X_train, X_people, X_test, y_train] = gd.get_mean_data(test_data = True);

save_lines = []
for i in range(len(X_test)):
    closest_people = cls.get_closest_people(X_test[i][0], X_train, y_train, X_people,1, distance = "braycurtis")
    save_lines.append(','.join([X_test[i][1], str(int(closest_people))]))

save_lines = list(sorted(save_lines))
save_lines.insert(0, "[track_id],[people_id]")
#Sauvegarde du fichier
with open('submit_file.csv', 'w') as f:
    f.write('\n'.join(save_lines))
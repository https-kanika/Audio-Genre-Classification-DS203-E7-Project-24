#chatgpt:
"""Implementation Strategy
First Step - MFCC Extraction: Begin by extracting MFCCs from each audio clip, if they aren’t already provided. Each MFCC vector will serve as an input to the CDBN model.
Training the CDBN: Use unsupervised learning on the MFCCs with the CDBN to capture meaningful hierarchical features. This involves training each RBM layer independently in a bottom-up fashion before fine-tuning the network.
Classification: After training, use the learned representations from the CDBN as input features for a classifier, such as an SVM, logistic regression, or a neural network, depending on the complexity and size of your dataset."""


#possible issues to keep in mind:-
# 1. training data too less for neural network if we are labeling data manually,
# to counter this we can collect a lot more songs and label them or use unsupervised learning in the end after advanced features are generated
# 2. feature loss in min-max pooling of mfcc
# 3.CDBN parameters need to be tuned


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

def preprocess_mfcc(mfccs):
    """
    Preprocess MFCC features.
    
    Parameters:
    mfccs (numpy.ndarray): The MFCC features.
    
    Returns:
    numpy.ndarray: The preprocessed MFCC features.
    """
    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs)
    mfccs_scaled = mfccs_scaled[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
    return mfccs_scaled

# Example usage
mfccs_preprocessed = preprocess_mfcc(mfccs)

def build_cdbn(input_shape):
    """
    Build a Convolutional Deep Belief Network (CDBN).
    
    Parameters:
    input_shape (tuple): Shape of the input data.
    
    Returns:
    tensorflow.keras.Model: The CDBN model.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))  # Adjust the number of classes as needed
    return model

import matplotlib.pyplot as plt

def visualize_filters(model, layer_name):
    """
    Visualize the filters of a convolutional layer.
    
    Parameters:
    model (tensorflow.keras.Model): The trained model.
    layer_name (str): The name of the convolutional layer.
    """
    layer = model.get_layer(name=layer_name)
    filters, biases = layer.get_weights()
    
    # Normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    n_filters = filters.shape[-1]
    fig, axes = plt.subplots(1, n_filters, figsize=(20, 5))
    
    for i in range(n_filters):
        f = filters[:, :, :, i]
        axes[i].imshow(f[:, :, 0], cmap='viridis')
        axes[i].axis('off')
    
    plt.show()

# Example usage


def visualize_feature_maps(model, layer_name, input_data):
    """
    Visualize the feature maps of a convolutional layer.
    
    Parameters:
    model (tensorflow.keras.Model): The trained model.
    layer_name (str): The name of the convolutional layer.
    input_data (numpy.ndarray): The input data.
    """
    layer = model.get_layer(name=layer_name)
    feature_map_model = models.Model(inputs=model.input, outputs=layer.output)
    feature_maps = feature_map_model.predict(input_data)
    
    n_features = feature_maps.shape[-1]
    fig, axes = plt.subplots(1, n_features, figsize=(20, 5))
    
    for i in range(n_features):
        axes[i].imshow(feature_maps[0, :, :, i], cmap='viridis')
        axes[i].axis('off')
    
    plt.show()

# Example usage




input_shape = mfccs_preprocessed.shape[1:]
cdbn_model = build_cdbn(input_shape)
cdbn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


X_train = np.array([mfccs_preprocessed])  # Replace with actual training data
y_train = np.array([0])  # Replace with actual labels

# Train the model
cdbn_model.fit(X_train, y_train, epochs=10, batch_size=1)

visualize_filters(cdbn_model, 'conv2d')
visualize_feature_maps(cdbn_model, 'conv2d', mfccs_preprocessed)

# Predict on new data
X_test = np.array([mfccs_preprocessed])  # Replace with actual test data
predictions = cdbn_model.predict(X_test)
print(predictions)


// ignore_for_file: unnecessary_const

import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:http/http.dart' as http;
import 'package:mess_i/utilities/helper_functions.dart';
import 'dart:convert';
import 'package:mess_i/utilities/widgets/date_picker.dart';
import 'package:mess_i/utilities/widgets/nav_bar.dart';
import 'package:hive/hive.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:intl/intl.dart';
import 'package:mess_i/config.dart';
import 'package:mess_i/models/student.dart';
import 'package:mess_i/utilities/auto_update.dart';

var tokenBox = Hive.box('token');
List<String> mealList = tokenBox.get('allowedMeals').split(",");

DateTime now = new DateTime.now();
DateTime dateToday = new DateTime(now.year, now.month, now.day);
DateTime date = dateToday;
bool showOfflineStats = true;
class MessSettingsPage extends StatefulWidget {
  @override
  State<StatefulWidget> createState() {
    return _MessSettingsPageState();
  }
}

class _MessSettingsPageState extends State<MessSettingsPage> {
  var studentBox;
  int rebateCount = 0;

  bool egg = false;
  bool milk = false;
  bool fruits = false;
  bool meal = true;
  String logic = "Logic 1";
  String hostel = "";
  String token = "";
  Map<String, dynamic> statsOut = {};
  late Map<String, int> mealCounters = {};


  String serverURL = CONFIG['serverUrl'];

  _MessSettingsPageState() {
    meal = mealList.contains("Meal");
    egg = mealList.contains("Egg");
    milk = mealList.contains("Milk");
    fruits = mealList.contains("Fruit");

    logic = tokenBox.get('logic');
    hostel = tokenBox.get('mess').split(",")[0];
  }

  Future<void> getMealStat(int y, int m, int d) async {
    String token = tokenBox.get('auth_token') ?? "no";
    String hostel = tokenBox.get('mess').split(",")[0] ?? "no";

    var mealCountersBox = await Hive.openBox<int>('mealCounters');
    DateTime now = DateTime.now();
    bool isToday = now.year == y && now.month == m && now.day == d;
    mealCounters = {
      'Breakfast': (isToday) ? tokenBox.get('breakfastCounter') ?? 0 : 0,
      'Lunch': (isToday) ? tokenBox.get('lunchCounter') ?? 0 : 0,
      'Snacks': (isToday) ? tokenBox.get('snacksCounter') ?? 0 : 0,
      'Dinner': (isToday) ? tokenBox.get('dinnerCounter') ?? 0 : 0,
      'Milk': (isToday) ? tokenBox.get('milkCounter') ?? 0 : 0,
      'Egg': (isToday) ? tokenBox.get('eggCounter') ?? 0 : 0,
      'Fruit': (isToday) ? tokenBox.get('fruitCounter') ?? 0 : 0,
    };

    statsOut={};
    final String url = "$serverURL/api/get-mess-data/$hostel/$y/$m/$d";

    var response =
    await http.get(Uri.parse(url), headers: {"x-access-token": token});
    if (response.statusCode == 200) {
      final Map<String, dynamic> data = jsonDecode(response.body);

      setState(() {
        for (var elem in data.entries) {
          statsOut[elem.key] = elem.value;
          if(elem.value > mealCounters[elem.key] && isToday){
            tokenBox.put(elem.key.toLowerCase()+"Counter",elem.value);
          }
        }
      });
    } else if (response.statusCode == 404 &&
        (response.reasonPhrase == "Data not available" ||
            response.reasonPhrase == "Not Found")) {
      statsOut = {};
    } else {
      statsOut = {};
      Fluttertoast.showToast(
        msg: "Error: ${response.reasonPhrase}",
        toastLength: Toast.LENGTH_SHORT,
        gravity: ToastGravity.CENTER,
      );
    }
  }

  void resetCounters() {
    DateTime lastUpdate = DateTime.fromMillisecondsSinceEpoch(tokenBox.get('lastUpdate') ?? 0);
    DateTime now = DateTime.now();

    if (lastUpdate.day != now.day || lastUpdate.month != now.month || lastUpdate.year != now.year) {
      tokenBox.put('breakfastCounter', 0);
      tokenBox.put('lunchCounter', 0);
      tokenBox.put('snacksCounter', 0);
      tokenBox.put('dinnerCounter', 0);
      tokenBox.put('milkCounter', 0);
      tokenBox.put('eggCounter', 0);
      tokenBox.put('fruitCounter', 0);
      tokenBox.put('lastUpdate', DateTime.now().millisecondsSinceEpoch);
    }
  }

  Future<void> getRebateStat() async {
    studentBox = await Hive.openBox<Student>('students');

    var now = DateTime.now();
    final studentList = studentBox.values
        .where((student) => student.rebate != null && student.rebate.isNotEmpty)
        .toList();

    if (studentList.isNotEmpty) {
      for (var i = 0; i < studentList.length; i++) {
        if (studentList[i].rebate!=null&&studentList[i].rebate.isNotEmpty) {
          for (final rebate in studentList[i].rebate) {
            if (rebate.isNotEmpty) {
              var inputFormat = DateFormat('dd-MM-yyyy');
              final start = inputFormat.parse(rebate[0]);
              final end = inputFormat.parse(rebate[1]);
              if (now.isAfter(start) && now.isBefore(end)) {
                rebateCount++;
                break;
              }
            }
          }
        }
      }
    }
  }


  @override
  void initState() {
    super.initState();
    date = dateToday;
    resetCounters();
    getMealStat(DateTime.now().year, DateTime.now().month, DateTime.now().day);
    getRebateStat();
  }

  @override
  Widget build(BuildContext context) {
    Dimensions getDims = Dimensions(context);

    final cards = <Widget>[];
    final List<String> meals = ['Breakfast', 'Lunch', 'Snacks', 'Dinner'];

    for (int i = 0; i < mealList.length; i += 1) {
      if (mealList[i] == 'Meal') {
        cards.add(Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              flex: 1,
              child: Container(
                margin: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: const Color(0xfffffcf4),
                  borderRadius: BorderRadius.all(Radius.circular(getDims.fractionHeight(0.022))),
                  border: Border.all(width: 2),
                ),
                // height: double.infinity,
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(getDims.fractionHeight(0.022)),
                  child: Image.asset('images/${mealList[i].toLowerCase()}-icon.png'),
                ),
              ),
            ),
            Expanded(
              flex: 5,
              child: Column(
                children: [
                  Row(
                    children: [
                      Text(
                        'Meals',
                        style: TextStyle(
                          fontSize: getDims.fractionWidth(0.015),
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const Divider(
                    color: Color(0xFFFFC42B),
                    thickness: 2,
                  ),
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Column(
                      children: meals.map((e) => Column(
                        children: [
                          Row(
                            children: [
                              Expanded(
                                flex: 1,
                                child: Text(
                                  e,
                                  style: TextStyle(
                                    fontSize: getDims.fractionWidth(0.015),
                                  ),
                                ),
                              ),
                              Expanded(
                                flex: 1,
                                child: Text(
                                  (showOfflineStats)?
                                    (statsOut[e] != null && statsOut[e] > mealCounters[e])
                                      ? '${statsOut[e]}'
                                      : '${mealCounters[e]}'
                                  : '${statsOut[e]??0}',
                                  style: TextStyle(
                                    fontSize: getDims.fractionWidth(0.015),
                                  ),
                                ),
                              ),
                            ],
                          ),
                          const Divider(
                            color: Color(0xFFFFC42B),
                            thickness: 1.2,
                          ),
                        ],
                      )).toList(),
                    ),
                  )
                ],
              ),
            ),
          ],
        ));
      } else {
        cards.add(Row(
          children: [
            Expanded(
              flex: 1,
              child: Container(
                margin: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: const Color(0xfffffcf4),
                  borderRadius: BorderRadius.all(Radius.circular(getDims.fractionHeight(0.022))),
                  border: Border.all(width: 2),
                ),
                // height: double.infinity,
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(getDims.fractionHeight(0.022)),
                  child: Image.asset('images/${mealList[i].toLowerCase()}-icon.png'),
                ),
              ),
            ),
            Expanded(
              flex: 5,
              child: Column(
                children: [
                  Row(
                      // mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Expanded(
                        flex: 1,
                        child: Text(
                          mealList[i],
                          style: TextStyle(
                            fontSize: getDims.fractionWidth(0.015),
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                      Expanded(
                        flex: 1,
                        child: Text(
                          (showOfflineStats)?
                          (statsOut[mealList[i]] != null && statsOut[mealList[i]] > mealCounters[mealList[i]])
                            ? '${statsOut[mealList[i]]}'
                            : '${mealCounters[mealList[i]]}'
                            : '${statsOut[mealList[i]]??0}',
                          style: TextStyle(
                            fontSize: getDims.fractionWidth(0.015),
                          ),
                        ),
                      )
                    ]),
                  const Divider(
                    color: Color(0xFFFFC42B),
                    thickness: 2,
                  )
                ],
              ),
            ),
          ],
        ));
      }
    }

    return Scaffold(
        backgroundColor: const Color(0xFFFFFCF4),
        body: Column(children: [
          Navbar(
            pageName: "SETTINGS",
            navButton: NavBarIconButton(
              navButtonIcon: const Icon(Icons.home_outlined),
              navButtonOnPressed: () {
                Navigator.pop(context);
                },
            )
          ),
          Container(
            height: getDims.fractionHeight(0.919),
            color: const Color(0xFFFFFCF4),
            child: Padding(
              padding: const EdgeInsets.all(8.0),
              child: Column(
                children: [
                  Expanded(
                    flex: 100,
                    child: Padding(
                      padding: EdgeInsets.symmetric(
                        horizontal: getDims.fractionWidth(0.04),
                      ),
                      child: Row(
                        children: [
                          Expanded(
                            child: Align(
                              alignment: Alignment.bottomLeft,
                              child: Text(
                                "Hostel: $hostel",
                                style: const TextStyle(
                                  fontSize: 40,
                                  fontFamily: 'OxygenBold',
                                ),
                              ),
                            ),
                          ),
                          const Padding(
                            padding: const EdgeInsets.all(8.0),
                            child: const SizedBox(
                              width:220,
                              child: Row( children: [
                                Align(
                                  alignment: Alignment.bottomRight,
                                  child:
                                    const Text(
                                      "Live count",
                                      textAlign : TextAlign.right,
                                      style: const TextStyle(
                                        fontSize: 32,
                                        fontFamily: 'Poppins',
                                        fontWeight: FontWeight.w600,
                                      ),
                                    )),
                                    Align(
                                      alignment: Alignment.centerRight,
                                      child: const Text(
                                        "(beta)",
                                        textAlign : TextAlign.right,
                                        style: const TextStyle(
                                          fontSize: 15,
                                          fontFamily: 'Poppins',
                                      ),
                                    ),)],
                                )
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.all(8.0),
                            child: SizedBox(
                              width: 40,
                              child: Align(
                                alignment: Alignment.bottomLeft,
                                child: Switch(
                                  value: showOfflineStats,
                                  activeTrackColor: const Color(0xffffc42b),
                                  inactiveTrackColor: const Color(0xffa8a8a8),
                                  thumbColor:MaterialStateProperty.resolveWith<Color?>((Set<MaterialState> states) {
                                    return const Color(0xfffffcf4);
                                  }),
                                  trackOutlineColor: MaterialStateProperty.resolveWith<Color?>((Set<MaterialState> states) {
                                    if (!showOfflineStats) {
                                      return const Color(0xffa8a8a8);
                                    } else {
                                      return const Color(0xfffffcf4);
                                    }
                                  }),

                                  onChanged: (bool value) {
                                    setState(() {
                                      showOfflineStats = value;
                                    });
                                  },
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  Divider(
                    indent: getDims.fractionWidth(0.045),
                    endIndent: getDims.fractionWidth(0.045),
                    color: const Color(0xFFFFC42B),
                    thickness: 5,
                  ),
                  Expanded(
                    flex: 574,
                    child: Padding(
                      padding: EdgeInsets.symmetric(
                        horizontal: getDims.fractionWidth(0.045),
                        vertical: getDims.fractionHeight(0.035)
                      ),
                      child: Row(
                        children: [
                          Expanded(
                            flex: 125,
                            child: Align(
                              alignment: Alignment.centerLeft,
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Expanded(
                                    flex: 1,
                                    child: Text(
                                      "Meal Stats",
                                      style: TextStyle(
                                        fontWeight: FontWeight.w500,
                                        fontSize: 35.6338,
                                      ),
                                    ),
                                  ),
                                  Expanded(
                                    flex: 5,
                                    child: Padding(
                                      padding: const EdgeInsets.all(8.0),
                                      child: Column(
                                        children: [
                                          Expanded(
                                            flex: 1,
                                            child: CustomDatePicker(
                                              popUpTitle: "SELECT DATE",
                                              name: "Date:",
                                              date: date,
                                              onDateTimeChanged: (DateTime newDate) {
                                                setState(() => date = newDate);
                                                setState(() {
                                                  getMealStat(date.year, date.month, date.day);
                                                });
                                              },
                                              context: context,
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ),
                                  Expanded(
                                    flex: 1,
                                    child: Column(
                                      children: [
                                        Container(
                                          height: getDims.fractionHeight(0.06),
                                          width: getDims.fractionWidth(0.16),
                                          decoration: BoxDecoration(
                                            boxShadow: [
                                              BoxShadow(
                                                offset: const Offset(13.5054, 17.7258),
                                                color: Colors.black.withOpacity(0.02),
                                                blurRadius: 8.44086
                                              ),
                                              BoxShadow(
                                                offset: const Offset(7.59677, 10.129),
                                                color: Colors.black.withOpacity(0.07),
                                                blurRadius: 7.59677
                                              ),
                                              BoxShadow(
                                                offset: const Offset(3.37634, 4.22043),
                                                color: Colors.black.withOpacity(0.11),
                                                blurRadius: 5.9086
                                              ),
                                              BoxShadow(
                                                offset: const Offset(0.844086, 0.844086),
                                                color: Colors.black.withOpacity(0.13),
                                                blurRadius: 3.37634
                                              ),
                                              // BoxShadow(offset: Offset(13.5054, 17.7258), color: Colors.black.withOpacity(0.02), blurRadius: 8.44086),
                                            ],
                                          ),
                                          child: Container(
                                            decoration: const BoxDecoration(
                                              color: const Color(0xff28282b),
                                              borderRadius: const BorderRadius.all(
                                                const Radius.circular(12),
                                              ),
                                            ),
                                            child: Center(
                                              child: Text(
                                                "Rebate Count: " + rebateCount.toString(),
                                                style: TextStyle(
                                                  color: Colors.white,
                                                  fontSize: getDims.fractionHeight(0.0285),
                                                ),
                                              ),
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                  Expanded(
                                    flex: 1,
                                    child: Column(
                                      children: [
                                        Container(
                                          height: getDims.fractionHeight(0.06),
                                          width: getDims.fractionWidth(0.16),
                                          decoration: BoxDecoration(
                                            boxShadow: [
                                              BoxShadow(
                                                offset: const Offset(13.5054, 17.7258),
                                                color: Colors.black.withOpacity(0.02),
                                                blurRadius: 8.44086
                                              ),
                                              BoxShadow(
                                                offset: const Offset(7.59677, 10.129),
                                                color: Colors.black.withOpacity(0.07),
                                                blurRadius: 7.59677
                                              ),
                                              BoxShadow(
                                                offset: const Offset(3.37634, 4.22043),
                                                color: Colors.black.withOpacity(0.11),
                                                blurRadius: 5.9086
                                              ),
                                              BoxShadow(
                                                offset: const Offset(0.844086, 0.844086),
                                                color: Colors.black.withOpacity(0.13),
                                                blurRadius: 3.37634
                                              ),
                                            ],
                                          ),
                                          child: TextButton(
                                            style: TextButton.styleFrom(
                                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12.0)),
                                              backgroundColor: const Color(0xff282828),
                                            ),
                                            child: Text(
                                              "Update App",
                                              style: TextStyle(
                                                color: Colors.white,
                                                fontSize: getDims.fractionHeight(0.0285)
                                              ),
                                            ),
                                            onPressed: () =>
                                                showDialog<String>(
                                              context: context,
                                              builder: (BuildContext context) => const UpdateScreen(),
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ]),
                          ),
                        ),
                        Expanded(
                          flex: 200,
                          child: SingleChildScrollView(
                            child: Column(
                              children: [
                                Row(
                                  children: [
                                    const Expanded(flex: 1, child: const SizedBox()),
                                    Expanded(
                                      flex: 5,
                                      child: Column(children: [
                                        Row(
                                          children: [
                                            Expanded(
                                              child: Align(
                                                alignment: Alignment.topLeft,
                                                child: Text(
                                                  "CARD",
                                                  style: TextStyle(
                                                    fontWeight: FontWeight.w700,
                                                    fontSize: getDims.fractionWidth(0.02),
                                                    color: const Color(0xffffc42b)
                                                  ),
                                                )
                                              ),
                                            ),
                                            Expanded(
                                              child: Align(
                                                alignment: Alignment.topLeft,
                                                child: Text(
                                                  "COUNT",
                                                  style: TextStyle(
                                                    fontWeight: FontWeight.w700,
                                                    fontSize: getDims.fractionWidth(0.02),
                                                    color: const Color(0xffffc42b)
                                                  ),
                                                )
                                              )
                                            ),
                                          ],
                                        ),
                                        const Divider(
                                          color: Color(0xFFFFC42B),
                                          thickness: 3,
                                        ),
                                      ]),
                                    ),
                                  ],
                                ),
                                Column(
                                  children: cards,
                                )
                              ],
                            ),
                          ),
                        ),],
                      ),
                    ),
                  ),
                  Expanded(
                    flex: 70,
                    child: Align(
                      alignment: Alignment.bottomLeft,
                      child: Text(
                        'v${CONFIG['version']}'
                      )
                    ),
                  )
                ],
              ),
            ),
          )
        ])
    );
  }
}

import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default='gt.txt', help='path to full log file')
args = parser.parse_args()

lines = open(args.path, 'r').readlines()

def EditDistDP(str1, str2):
	
	len1 = len(str1)
	len2 = len(str2)

	# Create a DP array to memoize result
	# of previous computations
	DP = [[0 for i in range(len1 + 1)] for j in range(2)]

	# Base condition when second String
	# is empty then we remove all characters
	for i in range(0, len1 + 1):
		DP[0][i] = i

	# Start filling the DP
	# This loop run for every
	# character in second String
	for i in range(1, len2 + 1):
		
		# This loop compares the char from
		# second String with first String
		# characters
		for j in range(0, len1 + 1):

			# If first String is empty then
			# we have to perform add character
			# operation to get second String
			if (j == 0):
				DP[i % 2][j] = i

			# If character from both String
			# is same then we do not perform any
			# operation . here i % 2 is for bound
			# the row number.
			elif(str1[j - 1] == str2[i-1]):
				DP[i % 2][j] = DP[(i - 1) % 2][j - 1]
			
			# If character from both String is
			# not same then we take the minimum
			# from three specified operation
			else:
				DP[i % 2][j] = (1 + min(DP[(i - 1) % 2][j],
									min(DP[i % 2][j - 1],
								DP[(i - 1) % 2][j - 1])))
			
	# After complete fill the DP array
	# if the len2 is even then we end
	# up in the 0th row else we end up
	# in the 1th row so we take len2 % 2
	# to get row
	return DP[len2 % 2][len1]



def word_acc(s1, s2):
    if s1 == s2:
        return True
    else:
        return False

def check_field(s1, s2, index):
    try:
        a = s1.split('/')[index]
        b = s2.split('/')[index]
        return 1 if a==b else 0
    except:
        return 0

def check_day(s1, s2):
    return check_field(s1, s2, 0)

def check_month(s1, s2):
    return check_field(s1, s2, 1)

def check_year(s1, s2):
    return check_field(s1, s2, 2)


gt_21 = []
pred_21 = []

gt_others = []
pred_others = []

for line in lines:
    flag = line.split()[0]
    gt = line.split(' ',1)[1].split(',')[0]
    pred = line.split(' ',1)[1].split(',')[1]
    year = gt.split('/')[-1]

    if (year == '21' or year == '2021' or year == '021'):
        gt_21.append(gt)
        pred_21.append(pred)

    else:
        gt_others.append(gt)
        pred_others.append(pred)

# print('total dates with year 21- ', len(gt_21))
# print('total dates with year < 21- ', len(gt_others))
# print('-'*20)

# ------------------------------------------


correct_21 = 0
ned_21 = 0
day_correct = 0
month_correct = 0
year_correct = 0

for i in range(len(gt_21)):
    if word_acc(gt_21[i], pred_21[i]):
        correct_21 += 1
    
    day_correct += check_day(gt_21[i], pred_21[i])
    month_correct += check_month(gt_21[i], pred_21[i])
    year_correct += check_year(gt_21[i], pred_21[i])

    ned_21  += EditDistDP(gt_21[i], pred_21[i]) / max(len(gt_21[i]), len(pred_21[i]))

len_gt21 = float(len(gt_21))

# if len(gt_21) != 0:

    # print('Word Acc for year 21 - ', correct_21/float(len(gt_21)))
    # print('Char Acc for year 21 - ', (float(len(gt_21))-ned_21)/float(len(gt_21)))
    # print('*'*20)
    # print('Day Acc for year 21 - ', day_correct/float(len(gt_21)))
    # print('Month Acc for year 21 - ', month_correct/float(len(gt_21)))
    # print('Year Acc for year 21 - ', year_correct/float(len(gt_21)))
    # print('-'*20)


# ------------------------------------

correct_others = 0
day_correct = 0
month_correct = 0
year_correct = 0
ned_others = 0

for i in range(len(gt_others)):
    if word_acc(gt_others[i], pred_others[i]):
        correct_others += 1
    
    day_correct += check_day(gt_others[i], pred_others[i])
    month_correct += check_month(gt_others[i], pred_others[i])
    year_correct += check_year(gt_others[i], pred_others[i])

    ned_others  += EditDistDP(gt_others[i], pred_others[i]) / max(len(gt_others[i]), len(pred_others[i]))

# print('Word Acc for year < 21 - ', correct_others/float(len(gt_others)))
# print('Char Acc for year < 21 - ', (len(gt_others)-ned_others)/float(len(gt_others)))
# print('*'*20)
# print('Day Acc for year < 21 - ', day_correct/float(len(gt_others)))
# print('Month Acc for year < 21 - ', month_correct/float(len(gt_others)))
# print('Year Acc for year < 21 - ', year_correct/float(len(gt_others)))
# print('-'*20)


# ------------------------------------



correct_all = 0
day_correct = 0
month_correct = 0
year_correct = 0
ned_all = 0
gt_all = gt_21 + gt_others
pred_all = pred_21 + pred_others

for i in range(len(gt_all)):
    if word_acc(gt_all[i], pred_all[i]):
        correct_all += 1
    
    day_correct += check_day(gt_all[i], pred_all[i])
    month_correct += check_month(gt_all[i], pred_all[i])
    year_correct += check_year(gt_all[i], pred_all[i])

    ned_all  += EditDistDP(gt_all[i], pred_all[i]) / max(len(gt_all[i]), len(pred_all[i]))


print('-'*20)

print('total images in dm- ', len(gt_all))
print('Overall Char Acc - ', (len(gt_all)-ned_all)/float(len(gt_all)))
print('Overall Word Acc- ', correct_all/float(len(gt_all)))
print('*'*20)
print('Overall Day Acc - ', day_correct/float(len(gt_all)))
print('Overall Month Acc - ', month_correct/float(len(gt_all)))
print('Overall Year Acc - ', year_correct/float(len(gt_all)))
print('-'*20)

    

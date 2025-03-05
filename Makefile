# ============================Table_general_case  non-iid ratio===========================
# ========================Cifar10========================
# --alpha 100
run20:
	python main.py --algo depthfl --data cifar10

run21:
	python main.py --algo splitmix --data cifar10

run22:
	python main.py --algo propose --data cifar10

run23:
	python main.py --algo depthfl --data cifar10 --client_split_ratios 1 0 0

# --alpha 1
run24:
	python main.py --algo depthfl --data cifar10 --alpha 1

run25:
	python main.py --algo splitmix --data cifar10 --alpha 1

run26:
	python main.py --algo propose --data cifar10 --alpha 1

run27:
	python main.py --algo depthfl --data cifar10 --client_split_ratios 1 0 0 --alpha 1

# --alpha 0.5
run28:
	python main.py --algo depthfl --data cifar10 --alpha 0.5

run29:
	python main.py --algo splitmix --data cifar10 --alpha 0.5

run30:
	python main.py --algo propose --data cifar10 --alpha 0.5

run31:
	python main.py --algo depthfl --data cifar10 --client_split_ratios 1 0 0 --alpha 0.5

# --alpha 0.25
run32:
	python main.py --algo depthfl --data cifar10 --alpha 0.25

run33:
	python main.py --algo splitmix --data cifar10 --alpha 0.25

run34:
	python main.py --algo propose --data cifar10 --alpha 0.25

run35:
	python main.py --algo depthfl --data cifar10 --client_split_ratios 1 0 0 --alpha 0.25

# --alpha 0.1
run36:
	python main.py --algo depthfl --data cifar10 --alpha 0.1

run37:
	python main.py --algo splitmix --data cifar10 --alpha 0.1

run38:
	python main.py --algo propose --data cifar10 --alpha 0.1

run39:
	python main.py --algo depthfl --data cifar10 --client_split_ratios 1 0 0 --alpha 0.1

# ========================Cifar100========================
# --alpha 100
run40:
	python main.py --algo depthfl --data cifar100

run41:
	python main.py --algo splitmix --data cifar100

run42:
	python main.py --algo propose --data cifar100

run43:
	python main.py --algo depthfl --data cifar100 --client_split_ratios 1 0 0

# --alpha 1
run44:
	python main.py --algo depthfl --data cifar100 --alpha 1

run45:
	python main.py --algo splitmix --data cifar100 --alpha 1

run46:
	python main.py --algo propose --data cifar100 --alpha 1

run47:
	python main.py --algo depthfl --data cifar100 --client_split_ratios 1 0 0 --alpha 1

# --alpha 0.5
run48:
	python main.py --algo depthfl --data cifar100 --alpha 0.5

run49:
	python main.py --algo splitmix --data cifar100 --alpha 0.5

run50:
	python main.py --algo propose --data cifar100 --alpha 0.5

run51:
	python main.py --algo depthfl --data cifar100 --client_split_ratios 1 0 0 --alpha 0.5

# --alpha 0.25
run52:
	python main.py --algo depthfl --data cifar100 --alpha 0.25

run53:
	python main.py --algo splitmix --data cifar100 --alpha 0.25

run54:
	python main.py --algo propose --data cifar100 --alpha 0.25

run55:
	python main.py --algo depthfl --data cifar100 --client_split_ratios 1 0 0 --alpha 0.25

# --alpha 0.1
run56:
	python main.py --algo depthfl --data cifar100 --alpha 0.1

run57:
	python main.py --algo splitmix --data cifar100 --alpha 0.1

run58:
	python main.py --algo propose --data cifar100 --alpha 0.1

run59:
	python main.py --algo depthfl --data cifar100 --client_split_ratios 1 0 0 --alpha 0.1

# ========================SVHN========================
# --alpha 100
run60:
	python main.py --algo depthfl --data svhn

run61:
	python main.py --algo splitmix --data svhn

run62:
	python main.py --algo propose --data svhn

run63:
	python main.py --algo depthfl --data svhn --client_split_ratios 1 0 0

# --alpha 1
run64:
	python main.py --algo depthfl --data svhn --alpha 1

run65:
	python main.py --algo splitmix --data svhn --alpha 1

run66:
	python main.py --algo propose --data svhn --alpha 1

run67:
	python main.py --algo depthfl --data svhn --client_split_ratios 1 0 0 --alpha 1

# --alpha 0.5
run68:
	python main.py --algo depthfl --data svhn --alpha 0.5

run69:
	python main.py --algo splitmix --data svhn --alpha 0.5

run70:
	python main.py --algo propose --data svhn --alpha 0.5

run71:
	python main.py --algo depthfl --data svhn --client_split_ratios 1 0 0 --alpha 0.5

# --alpha 0.25
run72:
	python main.py --algo depthfl --data svhn --alpha 0.25

run73:
	python main.py --algo splitmix --data svhn --alpha 0.25

run74:
	python main.py --algo propose --data svhn --alpha 0.25

run75:
	python main.py --algo depthfl --data svhn --client_split_ratios 1 0 0 --alpha 0.25

# --alpha 0.1
run76:
	python main.py --algo depthfl --data svhn --alpha 0.1

run77:
	python main.py --algo splitmix --data svhn --alpha 0.1

run78:
	python main.py --algo propose --data svhn --alpha 0.1

run79:
	python main.py --algo depthfl --data svhn --client_split_ratios 1 0 0 --alpha 0.1s


# ============================Table2 test weak client ratio cifar10===========================
run100:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.1 0 0.9

run101:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.1 0 0.9

run102:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.1 0 0.9

run103:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.2 0 0.8

run104:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.2 0 0.8

run105:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.2 0 0.8

run106:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.3 0 0.7

run107:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.3 0 0.7

run108:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.3 0 0.7

run109:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.4 0 0.6

run110:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.4 0 0.6

run111:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.4 0 0.6

run112:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.5 0 0.5

run113:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.5 0 0.5

run114:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.5 0 0.5

run115:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.6 0 0.4

run116:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.6 0 0.4

run117:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.6 0 0.4

run118:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.7 0 0.3

run119:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.7 0 0.3

run120:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.7 0 0.3

run121:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.8 0 0.2

run122:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.8 0 0.2

run123:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.8 0 0.2

run124:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.9 0 0.1

run125:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.9 0 0.1

run126:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.9 0 0.1

run127:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 1 0 0

run128:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0 0 1


# ============================Table2 test weak client ratio cifar100===========================
run130:
	python main.py --algo depthfl --data cifar100 --alpha 0.1 --client_split_ratios 0.1 0 0.9

run131:
	python main.py --algo splitmix --data cifar100 --alpha 0.1 --client_split_ratios 0.1 0 0.9

run132:
	python main.py --algo propose --data cifar100 --alpha 0.1 --client_split_ratios 0.1 0 0.9

run133:
	python main.py --algo depthfl --data cifar100 --alpha 0.1 --client_split_ratios 0.2 0 0.8

run134:
	python main.py --algo splitmix --data cifar100 --alpha 0.1 --client_split_ratios 0.2 0 0.8

run135:
	python main.py --algo propose --data cifar100 --alpha 0.1 --client_split_ratios 0.2 0 0.8

run136:
	python main.py --algo depthfl --data cifar100 --alpha 0.1 --client_split_ratios 0.3 0 0.7

run137:
	python main.py --algo splitmix --data cifar100 --alpha 0.1 --client_split_ratios 0.3 0 0.7

run138:
	python main.py --algo propose --data cifar100 --alpha 0.1 --client_split_ratios 0.3 0 0.7

run139:
	python main.py --algo depthfl --data cifar100 --alpha 0.1 --client_split_ratios 0.4 0 0.6

run140:
	python main.py --algo splitmix --data cifar100 --alpha 0.1 --client_split_ratios 0.4 0 0.6

run141:
	python main.py --algo propose --data cifar100 --alpha 0.1 --client_split_ratios 0.4 0 0.6

run142:
	python main.py --algo depthfl --data cifar100 --alpha 0.1 --client_split_ratios 0.5 0 0.5

run143:
	python main.py --algo splitmix --data cifar100 --alpha 0.1 --client_split_ratios 0.5 0 0.5

run144:
	python main.py --algo propose --data cifar100 --alpha 0.1 --client_split_ratios 0.5 0 0.5

run145:
	python main.py --algo depthfl --data cifar100 --alpha 0.1 --client_split_ratios 0.6 0 0.4

run146:
	python main.py --algo splitmix --data cifar100 --alpha 0.1 --client_split_ratios 0.6 0 0.4

run147:
	python main.py --algo propose --data cifar100 --alpha 0.1 --client_split_ratios 0.6 0 0.4

run148:
	python main.py --algo depthfl --data cifar100 --alpha 0.1 --client_split_ratios 0.7 0 0.3

run149:
	python main.py --algo splitmix --data cifar100 --alpha 0.1 --client_split_ratios 0.7 0 0.3

run150:
	python main.py --algo propose --data cifar100 --alpha 0.1 --client_split_ratios 0.7 0 0.3

run151:
	python main.py --algo depthfl --data cifar100 --alpha 0.1 --client_split_ratios 0.8 0 0.2

run152:
	python main.py --algo splitmix --data cifar100 --alpha 0.1 --client_split_ratios 0.8 0 0.2

run153:
	python main.py --algo propose --data cifar100 --alpha 0.1 --client_split_ratios 0.8 0 0.2

run154:
	python main.py --algo depthfl --data cifar100 --alpha 0.1 --client_split_ratios 0.9 0 0.1

run155:
	python main.py --algo splitmix --data cifar100 --alpha 0.1 --client_split_ratios 0.9 0 0.1

run156:
	python main.py --algo propose --data cifar100 --alpha 0.1 --client_split_ratios 0.9 0 0.1

run157:
	python main.py --algo depthfl --data cifar100 --alpha 0.1 --client_split_ratios 1 0 0

run158:
	python main.py --algo depthfl --data cifar100 --alpha 0.1 --client_split_ratios 0 0 1


# ============================Table2 test weak client ratio svhn===========================
run160:
	python main.py --algo depthfl --data svhn --alpha 0.1 --client_split_ratios 0.1 0 0.9

run161:
	python main.py --algo splitmix --data svhn --alpha 0.1 --client_split_ratios 0.1 0 0.9

run162:
	python main.py --algo propose --data svhn --alpha 0.1 --client_split_ratios 0.1 0 0.9

run163:
	python main.py --algo depthfl --data svhn --alpha 0.1 --client_split_ratios 0.2 0 0.8

run164:
	python main.py --algo splitmix --data svhn --alpha 0.1 --client_split_ratios 0.2 0 0.8

run165:
	python main.py --algo propose --data svhn --alpha 0.1 --client_split_ratios 0.2 0 0.8

run166:
	python main.py --algo depthfl --data svhn --alpha 0.1 --client_split_ratios 0.3 0 0.7

run167:
	python main.py --algo splitmix --data svhn --alpha 0.1 --client_split_ratios 0.3 0 0.7

run168:
	python main.py --algo propose --data svhn --alpha 0.1 --client_split_ratios 0.3 0 0.7

run169:
	python main.py --algo depthfl --data svhn --alpha 0.1 --client_split_ratios 0.4 0 0.6

run170:
	python main.py --algo splitmix --data svhn --alpha 0.1 --client_split_ratios 0.4 0 0.6

run171:
	python main.py --algo propose --data svhn --alpha 0.1 --client_split_ratios 0.4 0 0.6

run172:
	python main.py --algo depthfl --data svhn --alpha 0.1 --client_split_ratios 0.5 0 0.5

run173:
	python main.py --algo splitmix --data svhn --alpha 0.1 --client_split_ratios 0.5 0 0.5

run174:
	python main.py --algo propose --data svhn --alpha 0.1 --client_split_ratios 0.5 0 0.5

run175:
	python main.py --algo depthfl --data svhn --alpha 0.1 --client_split_ratios 0.6 0 0.4

run176:
	python main.py --algo splitmix --data svhn --alpha 0.1 --client_split_ratios 0.6 0 0.4

run177:
	python main.py --algo propose --data svhn --alpha 0.1 --client_split_ratios 0.6 0 0.4

run178:
	python main.py --algo depthfl --data svhn --alpha 0.1 --client_split_ratios 0.7 0 0.3

run179:
	python main.py --algo splitmix --data svhn --alpha 0.1 --client_split_ratios 0.7 0 0.3

run180:
	python main.py --algo propose --data svhn --alpha 0.1 --client_split_ratios 0.7 0 0.3

run181:
	python main.py --algo depthfl --data svhn --alpha 0.1 --client_split_ratios 0.8 0 0.2

run182:
	python main.py --algo splitmix --data svhn --alpha 0.1 --client_split_ratios 0.8 0 0.2

run183:
	python main.py --algo propose --data svhn --alpha 0.1 --client_split_ratios 0.8 0 0.2

run184:
	python main.py --algo depthfl --data svhn --alpha 0.1 --client_split_ratios 0.9 0 0.1

run185:
	python main.py --algo splitmix --data svhn --alpha 0.1 --client_split_ratios 0.9 0 0.1

run186:
	python main.py --algo propose --data svhn --alpha 0.1 --client_split_ratios 0.9 0 0.1

run187:
	python main.py --algo depthfl --data svhn --alpha 0.1 --client_split_ratios 1 0 0

run188:
	python main.py --algo depthfl --data svhn --alpha 0.1 --client_split_ratios 0 0 1

#===================both ratio and non-iid cifar10==========================
run200:
	python main.py --algo depthfl --data cifar10 --alpha 0.5 --client_split_ratios 0.7 0 0.3

run201:
	python main.py --algo splitmix --data cifar10 --alpha 0.5 --client_split_ratios 0.7 0 0.3

run202:
	python main.py --algo propose --data cifar10 --alpha 0.5 --client_split_ratios 0.7 0 0.3

run203:
	python main.py --algo depthfl --data cifar10 --alpha 0.5 --client_split_ratios 0.8 0 0.2

run204:
	python main.py --algo splitmix --data cifar10 --alpha 0.5 --client_split_ratios 0.8 0 0.2

run205:
	python main.py --algo propose --data cifar10 --alpha 0.5 --client_split_ratios 0.8 0 0.2

run206:
	python main.py --algo depthfl --data cifar10 --alpha 0.5 --client_split_ratios 0.9 0 0.1

run207:
	python main.py --algo splitmix --data cifar10 --alpha 0.5 --client_split_ratios 0.9 0 0.1

run208:
	python main.py --algo propose --data cifar10 --alpha 0.5 --client_split_ratios 0.9 0 0.1

run209:
	python main.py --algo depthfl --data cifar10 --alpha 0.5 --client_split_ratios 1 0 0

run210:
	python main.py --algo depthfl --data cifar10 --alpha 0.25 --client_split_ratios 0.7 0 0.3

run211:
	python main.py --algo splitmix --data cifar10 --alpha 0.25 --client_split_ratios 0.7 0 0.3

run212:
	python main.py --algo propose --data cifar10 --alpha 0.25 --client_split_ratios 0.7 0 0.3

run213:
	python main.py --algo depthfl --data cifar10 --alpha 0.25 --client_split_ratios 0.8 0 0.2

run214:
	python main.py --algo splitmix --data cifar10 --alpha 0.25 --client_split_ratios 0.8 0 0.2

run215:
	python main.py --algo propose --data cifar10 --alpha 0.25 --client_split_ratios 0.8 0 0.2

run216:
	python main.py --algo depthfl --data cifar10 --alpha 0.25 --client_split_ratios 0.9 0 0.1

run217:
	python main.py --algo splitmix --data cifar10 --alpha 0.25 --client_split_ratios 0.9 0 0.1

run218:
	python main.py --algo propose --data cifar10 --alpha 0.25 --client_split_ratios 0.9 0 0.1

run219:
	python main.py --algo depthfl --data cifar10 --alpha 0.25 --client_split_ratios 1 0 0

#===================both ratio and non-iid svhn==========================
run220:
	python main.py --algo depthfl --data svhn --alpha 0.5 --client_split_ratios 0.7 0 0.3

run221:
	python main.py --algo splitmix --data svhn --alpha 0.5 --client_split_ratios 0.7 0 0.3

run222:
	python main.py --algo propose --data svhn --alpha 0.5 --client_split_ratios 0.7 0 0.3

run223:
	python main.py --algo depthfl --data svhn --alpha 0.5 --client_split_ratios 0.8 0 0.2

run224:
	python main.py --algo splitmix --data svhn --alpha 0.5 --client_split_ratios 0.8 0 0.2

run225:
	python main.py --algo propose --data svhn --alpha 0.5 --client_split_ratios 0.8 0 0.2

run226:
	python main.py --algo depthfl --data svhn --alpha 0.5 --client_split_ratios 0.9 0 0.1

run227:
	python main.py --algo splitmix --data svhn --alpha 0.5 --client_split_ratios 0.9 0 0.1

run228:
	python main.py --algo propose --data svhn --alpha 0.5 --client_split_ratios 0.9 0 0.1

run229:
	python main.py --algo depthfl --data svhn --alpha 0.5 --client_split_ratios 1 0 0

run230:
	python main.py --algo depthfl --data svhn --alpha 0.25 --client_split_ratios 0.7 0 0.3

run231:
	python main.py --algo splitmix --data svhn --alpha 0.25 --client_split_ratios 0.7 0 0.3

run232:
	python main.py --algo propose --data svhn --alpha 0.25 --client_split_ratios 0.7 0 0.3

run233:
	python main.py --algo depthfl --data svhn --alpha 0.25 --client_split_ratios 0.8 0 0.2

run234:
	python main.py --algo splitmix --data svhn --alpha 0.25 --client_split_ratios 0.8 0 0.2

run235:
	python main.py --algo propose --data svhn --alpha 0.25 --client_split_ratios 0.8 0 0.2

run236:
	python main.py --algo depthfl --data svhn --alpha 0.25 --client_split_ratios 0.9 0 0.1

run237:
	python main.py --algo splitmix --data svhn --alpha 0.25 --client_split_ratios 0.9 0 0.1

run238:
	python main.py --algo propose --data svhn --alpha 0.25 --client_split_ratios 0.9 0 0.1

run239:
	python main.py --algo depthfl --data svhn --alpha 0.25 --client_split_ratios 1 0 0

#===================both ratio and non-iid cifar100==========================
run240:
	python main.py --algo depthfl --data cifar100 --alpha 0.5 --client_split_ratios 0.7 0 0.3

run241:
	python main.py --algo splitmix --data cifar100 --alpha 0.5 --client_split_ratios 0.7 0 0.3

run242:
	python main.py --algo propose --data cifar100 --alpha 0.5 --client_split_ratios 0.7 0 0.3

run243:
	python main.py --algo depthfl --data cifar100 --alpha 0.5 --client_split_ratios 0.8 0 0.2

run244:
	python main.py --algo splitmix --data cifar100 --alpha 0.5 --client_split_ratios 0.8 0 0.2

run245:
	python main.py --algo propose --data cifar100 --alpha 0.5 --client_split_ratios 0.8 0 0.2

run246:
	python main.py --algo depthfl --data cifar100 --alpha 0.5 --client_split_ratios 0.9 0 0.1

run247:
	python main.py --algo splitmix --data cifar100 --alpha 0.5 --client_split_ratios 0.9 0 0.1

run248:
	python main.py --algo propose --data cifar100 --alpha 0.5 --client_split_ratios 0.9 0 0.1

run249:
	python main.py --algo depthfl --data cifar100 --alpha 0.5 --client_split_ratios 1 0 0

run250:
	python main.py --algo depthfl --data cifar100 --alpha 0.25 --client_split_ratios 0.7 0 0.3

run251:
	python main.py --algo splitmix --data cifar100 --alpha 0.25 --client_split_ratios 0.7 0 0.3

run252:
	python main.py --algo propose --data cifar100 --alpha 0.25 --client_split_ratios 0.7 0 0.3

run253:
	python main.py --algo depthfl --data cifar100 --alpha 0.25 --client_split_ratios 0.8 0 0.2

run254:
	python main.py --algo splitmix --data cifar100 --alpha 0.25 --client_split_ratios 0.8 0 0.2

run255:
	python main.py --algo propose --data cifar100 --alpha 0.25 --client_split_ratios 0.8 0 0.2

run256:
	python main.py --algo depthfl --data cifar100 --alpha 0.25 --client_split_ratios 0.9 0 0.1

run257:
	python main.py --algo splitmix --data cifar100 --alpha 0.25 --client_split_ratios 0.9 0 0.1

run258:
	python main.py --algo propose --data cifar100 --alpha 0.25 --client_split_ratios 0.9 0 0.1

run259:
	python main.py --algo depthfl --data cifar100 --alpha 0.25 --client_split_ratios 1 0 0


#========================client ratio random cifar10========================
run300:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.1 0 0.9

run301:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.2 0 0.8 

run302:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.3 0 0.7 

run303:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.4 0 0.6 

run304:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.5 0 0.5 

run305:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.6 0 0.4

run306:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.7 0 0.3 

run307:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.8 0 0.2

run308:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0.9 0 0.1

run309:
	python main.py --algo random --data cifar10 --alpha 0.1 --client_split_ratios 0.1 0 0.9

run310:
	python main.py --algo random --data cifar10 --alpha 0.1 --client_split_ratios 0.2 0 0.8 

run311:
	python main.py --algo random --data cifar10 --alpha 0.1 --client_split_ratios 0.3 0 0.7 

run312:
	python main.py --algo random --data cifar10 --alpha 0.1 --client_split_ratios 0.4 0 0.6 

run313:
	python main.py --algo random --data cifar10 --alpha 0.1 --client_split_ratios 0.5 0 0.5

run314:
	python main.py --algo random --data cifar10 --alpha 0.1 --client_split_ratios 0.6 0 0.4

run315:
	python main.py --algo random --data cifar10 --alpha 0.1 --client_split_ratios 0.7 0 0.3 

run316:
	python main.py --algo random --data cifar10 --alpha 0.1 --client_split_ratios 0.8 0 0.2 

run317:
	python main.py --algo random --data cifar10 --alpha 0.1 --client_split_ratios 0.9 0 0.1 

run318:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.1 0 0.9

run319:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.2 0 0.8

run320:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.3 0 0.7

run321:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.4 0 0.6

run322:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.5 0 0.5

run323:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.6 0 0.4

run324:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.7 0 0.3

run325:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.8 0 0.2

run326:
	python main.py --algo propose --data cifar10 --alpha 0.1 --client_split_ratios 0.9 0 0.1

run327:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 1 0 0

run328:
	python main.py --algo depthfl --data cifar10 --alpha 0.1 --client_split_ratios 0 0 1

run329:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.1 0 0.9

run330:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.2 0 0.8

run331:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.3 0 0.7

run332:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.4 0 0.6

run333:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.5 0 0.5

run334:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.6 0 0.4

run335:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.7 0 0.3

run336:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.8 0 0.2

run337:
	python main.py --algo splitmix --data cifar10 --alpha 0.1 --client_split_ratios 0.9 0 0.1


# ==============Fedavg(Large)=================
run80:
	python main.py --algo depthfl --data cifar10 --client_split_ratios 0 0 1 --alpha 0.5

run81:
	python main.py --algo depthfl --data cifar10 --client_split_ratios 0 0 1 --alpha 0.25

run82:
	python main.py --algo depthfl --data cifar100 --client_split_ratios 0 0 1 --alpha 0.5

run83:
	python main.py --algo depthfl --data cifar100 --client_split_ratios 0 0 1 --alpha 0.25


all: run322 run319
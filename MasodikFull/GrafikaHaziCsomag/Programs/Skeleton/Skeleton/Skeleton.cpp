//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kaszala Kristof
// Neptun : S9XEU5
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================

#include "framework.h"

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t; // csak ha pozitív = eltaláltál valamit 
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

struct Cylinder : public Intersectable {
	vec3 center;
	float radius;

	Cylinder(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		float sx, sy, sz, dx, dy, dz, cx, cy, cz;
		sx = ray.start.x;
		sy = ray.start.y;
		sz = ray.start.z;
		dx = ray.dir.x;
		dy = ray.dir.y;
		dz = ray.dir.z;
		cx = center.x;
		cy = center.y;
		cz = center.z;

		float xnc = 0.9968018, ync = 0.99764933, znc = 0.0055042;
		float xy = 0.00532548, xz = 0.11262136, yz = 0.09598324;
		float xc = 1.9936036, yc = -0.00532548, zc = 0.11262136;

		float a = xnc * powf(dx, 2.0)
			+ ync * powf(dy, 2.0f)
			+ znc * powf(dz, 2.0f)
			+ xy * dx * dy
			- xz * dx * dz
			+ yz * dy * dz;
		float b = 2.0f * xnc * dx * (sx - cx)
			+ 2.0f * ync * dy * (sy - cy)
			+ 2.0f * znc * dz * (sz - cz)
			+ xy * (sx * dy + dx * sy - dx * cy - cx * dy)
			- xz * (sx * dz + dx * sz - dx * cz - cx * dz)
			+ yz * (sy * dz + dy * sz - dy * cz - cy * dz)
			- xc * dx
			- yc * dy
			+ zc * dz;
		float c = xnc * powf(sx - cx, 2.0f)
			+ ync * powf(sy - cy, 2.0f)
			+ znc * powf(sz - cz, 2.0f)
			+ xy * (sx * sy - sx * cy - cx * sy + cx * cy)
			- xz * (sx * sz - sx * cz - cx * sz + cx * cz)
			+ yz * (sy * sz - sy * cz - cy * sz + cy * cz)
			- xc * (sx - cx)
			- yc * (sy - cy)
			+ zc * (sz - cz) - 0.0031982f;

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;

		hit.normal.x = 2.0f * xnc * (hit.position.x - center.x) + xy * (hit.position.y - center.y) - xz * (hit.position.z - center.z) - xc;
		hit.normal.y = xy * (hit.position.x - center.x) + 2.0f * ync * (hit.position.y - center.y) + yz * (hit.position.z - center.z) - yc;
		hit.normal.z = -xz * (hit.position.x - center.x) + 2.0f * znc * (hit.position.z - center.z) + yz * (hit.position.y - center.y) + zc;

		hit.normal = normalize(hit.normal);
		hit.material = material;
		return hit;
	}
};

struct Cylinder2 : public Intersectable {
	vec3 center;
	float radius;

	Cylinder2(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		float a = powf(ray.dir.x, 2.0f) + pow(ray.dir.y, 2.0f); //
		float b = 2.0f * (
			(ray.start.x - center.x) * ray.dir.x
			+ (ray.start.y - center.y) * ray.dir.y
			);
		float c = powf(ray.start.x - center.x, 2.0f) + powf(ray.start.y - center.y, 2.0f) - 1.0f;

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;

		hit.normal.x = 2.0f * (hit.position.x - center.x);//
		hit.normal.y = 2.0f * (hit.position.y - center.y);
		hit.normal.z = 0.0f;

		hit.material = material;
		hit.normal = normalize(hit.normal);
		return hit;
	}
};




struct Hyperboloid : public Intersectable {
	vec3 center;
	float radius;

	Hyperboloid(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		float sx, sy, sz, dx, dy, dz, cx, cy, cz;
		sx = ray.start.x;
		sy = ray.start.y;
		sz = ray.start.z;
		dx = ray.dir.x;
		dy = ray.dir.y;
		dz = ray.dir.z;
		cx = center.x;
		cy = center.y;
		cz = center.z;

		//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		float xnc = 0.155745, ync = 0.797594, znc = 0.880816;
		float xy = 0.130065, xz = 0.880816, yz = 1.010843;
		float xc = 0.31149, yc = 0.130065, zc = 0.880816;

		float a = xnc * powf(dx, 2.0)
			+ ync * powf(dy, 2.0f)
			- znc * powf(dz, 2.0f)
			- xy * dx * dy
			- xz * dx * dz
			+ yz * dy * dz;
		float b = 2.0f * xnc * dx * (sx - cx)
			+ 2.0f * ync * dy * (sy - cy)
			- 2.0f * znc * dz * (sz - cz)
			- xy * (sx * dy + dx * sy - dx * cy - cx * dy)
			- xz * (sx * dz + dx * sz - dx * cz - cx * dz)
			+ yz * (sy * dz + dy * sz - dy * cz - cy * dz)
			- xc * dx
			+ yc * dy
			+ zc * dz;
		float c = xnc * powf(sx - cx, 2.0f)
			+ ync * powf(sy - cy, 2.0f)
			- znc * powf(sz - cz, 2.0f)
			- xy * (sx * sy - sx * cy - cx * sy + cx * cy)
			+ yz * (sy * sz - sy * cz - cy * sz + cy * cz)
			- xz * (sx * sz - sx * cz - cx * sz + cx * cz)
			- xc * (sx - cx)
			+ yc * (sy - cy)
			+ zc * (sz - cz) - 0.8442548f;

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;

		hit.normal.x = 2.0f * xnc * (hit.position.x - center.x) - xy * (hit.position.y - center.y) - xz * (hit.position.z - center.z) - xc;
		hit.normal.y = -1.0f * xy * (hit.position.x - center.x) + 2.0f * ync * (hit.position.y - center.y) + yz * (hit.position.z - center.z) + yc;
		hit.normal.z = -1.0f * xz * (hit.position.x - center.x) - 2.0f * znc * (hit.position.z - center.z) + yz * (hit.position.y - center.y) + zc;


		hit.normal = normalize(hit.normal);
		hit.material = material;
		return hit;
		//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

				//float xnc = 0.99742497, ync = 0.99817267, znc = 0.99516096;
				//float xy = 0.00437608, xz = 0.14446988, yz = 0.12204006;
				//float xc = 0.14448436, yc = 0.12205226, zc = 1.99072148;//

				//float a = xnc * powf(dx, 2.0) + ync * powf(dy, 2.0f) - znc * powf(dz, 2.0f) 
				//		+ xy *dx*dy 
				//	    - xz*dx*dz 
				//	    + yz*dy*dz;
				//float b = 2.0f * xnc * dx*(sx - cx) + 2.0f * ync * dy*(sy - cy) - 2.0f*znc * dz*(sz - cz)
				//		+ xy * (sx*dy + dx * sy - dx * cy - cx * dy)
				//		- xz * (sx*dz + dx * sz - dx * cz - cx * dz)
				//		+ yz * (sy*dz + dy * sz - dy * cz - cy * dz)
				//		+ xc * dx 
				//		- yc * dy 
				//		+ zc * dz;

				//float c = xnc * powf(sx - cx, 2.0f) + ync * powf(sy - cy, 2.0f) - znc * powf(sz - cz, 2.0f)
				//		+ xy * (sx*sy - sx * cy - cx * sy + cx * cy)
				//		- xz * (sx*sz - sx * cz - cx * sz + cx * cz)
				//		+ yz * (sy*sz - sy * cz - cy * sz + cy * cz)
				//		+ xc * (sx - cx)
				//		- yc * (sy - cy)
				//		+ zc * (sz - cz) - 1.99556052f;
				//		

				//float discr = b * b - 4.0f * a * c;
				//if (discr < 0) return hit;
				//float sqrt_discr = sqrtf(discr);
				//float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
				//float t2 = (-b - sqrt_discr) / 2.0f / a;
				//if (t1 <= 0) return hit;
				//hit.t = (t2 > 0) ? t2 : t1;
				//hit.position = ray.start + ray.dir * hit.t;

				//hit.normal.x = 2.0f * xnc *(hit.position.x - center.x) + xy * (hit.position.y - center.y) - xz * (hit.position.z - center.z) + xc;
				//hit.normal.y = xy * (hit.position.x - center.x) + 2.0f * ync * (hit.position.y - center.y) + yz * (hit.position.z - center.z) - yc;
				//hit.normal.z = -1.0f*xz * (hit.position.x - center.x) - 2.0f * znc * (hit.position.z - center.z) + yz * (hit.position.y - center.y) + zc;


				//hit.normal = normalize(hit.normal);
				//hit.material = material;
				//return hit;
	}
};

struct Hyperboloid2 : public Intersectable {     /// forgatás nélküli hiperboloid
	vec3 center;
	float radius;

	Hyperboloid2(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		//-----------------------------------------------------FORGATÁS NÉLKÜLI------------------------------------------------------------------------	
		float a = powf(ray.dir.x, 2.0f) + powf(ray.dir.y, 2.0f) - powf(ray.dir.z, 2.0f);
		float b = 2.0f * (
			(ray.start.x - center.x) * ray.dir.x
			+ (ray.start.y - center.y) * ray.dir.y
			- (ray.start.z - center.z) * ray.dir.z
			);
		float c = powf(ray.start.x - center.x, 2.0f) + powf(ray.start.y - center.y, 2.0f) - powf(ray.start.z - center.z, 2.0f) - 1.0f;

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;

		hit.normal.x = (2.0f * (hit.position.x - center.x));
		hit.normal.y = (2.0f * (hit.position.y - center.y));
		hit.normal.z = (-2.0f * (hit.position.z - center.z));

		hit.normal = normalize(hit.normal);
		hit.material = material;
		return hit;
	}
};

const int nTesselatedVertices = 100;
struct Curve {
	std::vector<vec4> wCtrlPoints;		// coordinates of control points
	virtual vec4 r(float t) { return wCtrlPoints[0]; }
	float tStart() { return 0; }
	float tEnd() { return 1; }

	std::vector<vec4> vertexData;

	void AddControlPoint(vec2 cPoint) {
		vec4 wVertex = vec4(cPoint.x, cPoint.y, 0, 0);
		wCtrlPoints.push_back(wVertex);
	}

	void Calculate() {
		for (int i = 0; i < nTesselatedVertices; i++) {	// Tessellate
			float tNormalized = (float)i / (nTesselatedVertices - 1);
			float t = tStart() + (tEnd() - tStart()) * tNormalized;
			vec4 wVertex = r(t);
			vertexData.push_back(vec4(wVertex.x, wVertex.y, wVertex.z, 1.0));
		}
	}
};

//Bezier using Bernstein polynomials
struct BezierCurve : public Curve {
	float B(int i, float t) {
		int n = wCtrlPoints.size() - 1; // n deg polynomial = n+1 pts!
		float choose = 1;
		for (int j = 1; j <= i; j++) choose *= (float)(n - j + 1) / j;
		return choose * pow(t, i) * pow(1 - t, n - i);
	}
	vec4 r(float t) {
		vec4 wPoint = vec4(0, 0, 0, 0);
		for (unsigned int n = 0; n < wCtrlPoints.size(); n++) wPoint += wCtrlPoints[n] * B(n, t);
		return wPoint;
	}
};


class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 point;
	vec3 Le;
	float intensity; // majd az interferenciánál

	Light(vec3 _point, vec3 _Le, float _intensity) {
		point = _point;
		Le = _Le;
		intensity = _intensity;
	}
	float getIntensity(float dist) {
		if (dist < 1.0f) return intensity;
		return intensity * (1.0f / (dist * dist));
	}
};
const float epsilon = 0.0001f;
struct Light2 {
	vec3 location;
	vec3 power;

	Light2(vec3 _location, vec3 _power) {
		location = _location;
		power = _power;
	}
	double distanceOf(vec3 point) {
		return length(location - point);
	}
	vec3 directionOf(vec3 point) {
		return normalize(location - point);
	}
	vec3 radianceAt(vec3 point) {
		double distance2 = dot(location - point, location - point);
		if (distance2 < epsilon) distance2 = epsilon;
		return power / distance2 / 4 / M_PI;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	std::vector<Light2*> lights2;
	Camera camera;
	vec3 La;
public:
	void build() {
		std::vector<vec4> points;
		std::vector<vec4> points2;
		BezierCurve bezier;

		bezier.AddControlPoint(vec2(0.2617993, 1.0));
		bezier.AddControlPoint(vec2(M_PI / 2, 50.0));
		bezier.AddControlPoint(vec2(3.14159, 1.0));

		bezier.Calculate();
		for (int i = 0; i < nTesselatedVertices; i++) {
			//printf("x:%f y:%f\n", bezier.vertexData[i].x, bezier.vertexData[i].y);
			float x, y, z;
			x = cos(bezier.vertexData[i].x);
			y = sin(bezier.vertexData[i].x);
			z = y;
			points.push_back(vec4(x, y, z, 1.0));
		}

		// ez jött ki pap?n a henger transzform????
		mat4 srt(
			0.986189, 0.158452, -0.048214, 0.0,
			-0.155690, 0.986189, 0.056500, 0.0,
			0.169501, -0.144641, 2.991714, 0.0,
			1.0, 0.0, 0.0, 1.0
		);
		// uj vektorba a transzformalt U pontjai, (hogy ki lehessen rajzolni mind2ot)
		for (int i = 0; i < nTesselatedVertices; i++) {
			vec4 asd = points[i] * srt;
			points2.push_back(asd);
		}

		//mat4 scaled = ScaleMatrix(vec3(1.0f, 1.0f, 3.0f));         // Transzform???üggv?ek
		//mat4 rotated = RotationMatrix(0.174532925f, vec3(1.0f, 1.0f, 3.0f));   // 0.5235987756 rad = 30 deg     0.174532925 = 10 deg
		//mat4 translated = TranslateMatrix(vec3(1.0f, 0.0f, 0.0f));

//-------------------------------------------------------------------------Jo kamerak + fenyek ---------------------------------------

		//vec3 eye = vec3(10, -10, -4); vec3 vup = vec3(1, 1, 3); vec3 lookat = vec3(3, 4, 5); 
		//lights.push_back(new Light(vec3(50.0f, -70.5f, -10.6f), vec3(2, 2, 2), 30.0f));

		//vec3 eye = vec3(18, -7, -10); vec3 vup = vec3(1, -1.4, 3); vec3 lookat = vec3(-1, 1, 2.8);
		//lights.push_back(new Light(vec3(8.0f, -8.0, -5.0f), vec3(2, 2, 2), 30.0f));   //

		vec3 eye = vec3(10, -10, -4); vec3 vup = vec3(0, 1, 0); vec3 lookat = vec3(3, 4, 5);
		lights.push_back(new Light(vec3(8.0f, -8.0, -5.0f), vec3(2, 2, 2), 30.0f));   //

		//vec3 eye = vec3(18, -7, -10); vec3 vup = vec3(1, -1.4, 3); vec3 lookat = vec3(-1, 1, 2.8);
		//lights.push_back(new Light(vec3(4.4f, -5.0f, 7.6f), vec3(1, 1, 2), 30.0f));


		//lights2.push_back(new Light2(vec3(4.4f, -5.0f, 7.6f), vec3(1500, 1500, 1500)));
		//lights2.push_back(new Light2(vec3(2, 2, 3), vec3(500, 500, 500)));

		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		La = vec3(0.1f, 0.1f, 0.1f);
		vec3 kd(0.13f, 0.13f, 0.33f), ks(2, 3, 5);
		Material* material = new Material(kd, ks, 40);

		//------------------------------------------Cylinder-------------------------------------------------------------------
		objects.push_back(new Cylinder(vec3(0.0f, 0.0f, 0.0f), 0.0f, material)); // transzformalt henger
		//objects.push_back(new Cylinder2(vec3(0.0f, 0.0f,0.0f), 0.0f, material));  // eredeti henger

//------------------------------------------Hiperboloid----------------------------------------------------------------
		objects.push_back(new Hyperboloid(vec3(2.5f, 10.0f, 5.5f), 0.0f, material)); // transzformalt hyperboloid
		//objects.push_back(new Hyperboloid2(vec3(0.0f, 0.0f, 4.0f), 0.0f, material));  // eredeti hiperboloid

//------------------------------------------U alak---------------------------------------------------------------------
		/*for (int i = 0; i < nTesselatedVertices; i++) {   // transzformacio nelkuli U
			objects.push_back(new Sphere(vec3(points[i].x,points[i].y,points[i].z), 0.03f, material));
		}*/
		/*for (int i = 0; i < nTesselatedVertices; i++) {   // transzformalt U betu kirajzolasa gombokbol
			objects.push_back(new Sphere(vec3(points2[i].x, points2[i].y, points2[i].z), 0.05f, new Material(vec3(0.8, 0.00, 0.00), ks, 70)));
		}*/

		//------------------------------------------U pontokbol pontfeny-------------------------------------------------------
		for (int i = 0; i < nTesselatedVertices; i++) {
			lights2.push_back(new Light2(vec3(points2[i].x, points2[i].y, points2[i].z), vec3(0, 300, 0)));
		}

		/*for (int i = 0; i < nTesselatedVertices; i++) {
			lights.push_back(new Light(vec3(points2[i].x, points2[i].y, points2[i].z), vec3(1, 1, 2), 30.0f));
		}*/

	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		vec3 sourcePoint; // ahonnan jön a fény
		for (Light* light : lights) {
			sourcePoint = normalize(light->point - hit.position);
			Ray shadowRay(hit.position + hit.normal * epsilon, sourcePoint);
			float cosTheta = dot(hit.normal, normalize(sourcePoint));
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + sourcePoint);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}

		int i = 0;
		vec3 outDir;
		vec3 N = hit.normal;	// normal of the visible surface
		for (auto light : lights2) {	// Direct light source computation
			outDir = light->directionOf(hit.position);
			Hit shadowHit = firstIntersect(Ray(hit.position + N * epsilon, outDir));
			if (shadowHit.t < epsilon || shadowHit.t > light->distanceOf(hit.position)) {	// if not in shadow
				float cosThetaL = dot(N, outDir);
				if (cosThetaL >= epsilon) {
					float maxFazis = 0.65;
					float fazis = findMod(light->distanceOf(hit.position), maxFazis);
					if (maxFazis - fazis > maxFazis / 2.0f) {
						outRadiance = outRadiance + hit.material->kd / M_PI * cosThetaL * light->radianceAt(hit.position);
					}
					else outRadiance = outRadiance - hit.material->kd / M_PI * cosThetaL * light->radianceAt(hit.position);
				}
			}
		}
		return outRadiance;
	}

	float findMod(float a, float b)
	{
		float mod;

		if (a < 0)
			mod = -a;
		else
			mod = a;
		if (b < 0)
			b = -b;

		while (mod >= b)
			mod = mod - b;

		if (a < 0)
			return -mod;

		return mod;
	}
};

// innen nem kell hozzányúlni
GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
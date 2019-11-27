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
// Nev    : 
// Neptun : 
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
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t; // csak ha pozitív = eltaláltál valamit 
	vec3 position, normal;
	Material * material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material * material;
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
		float b = 2.0f*xnc * dx*(sx - cx)
				+ 2.0f*ync * dy*(sy - cy)
				+ 2.0f*znc * dz*(sz - cz)
				+ xy * (sx*dy + dx * sy - dx * cy - cx * dy)
				- xz * (sx*dz + dx * sz - dx * cz - cx * dz)
				+ yz * (sy*dz + dy * sz - dy * cz - cy * dz)
				- xc * dx
				- yc * dy
				+ zc * dz;
		float c = xnc * powf(sx - cx, 2.0f)
				+ ync * powf(sy - cy, 2.0f)
				+ znc * powf(sz - cz, 2.0f)
				+ xy * (sx*sy - sx * cy - cx * sy + cx * cy)
				- xz * (sx*sz - sx * cz - cx * sz + cx * cz)
				+ yz * (sy*sz - sy * cz - cy * sz + cy * cz)
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

		hit.normal.x = 2.0f * xnc *(hit.position.x - center.x) + xy * (hit.position.y - center.y) - xz * (hit.position.z - center.z) - xc;
		hit.normal.y = xy * (hit.position.x - center.x) + 2.0f * ync * (hit.position.y - center.y) + yz * (hit.position.z - center.z) - yc;
		hit.normal.z = - xz * (hit.position.x - center.x) + 2.0f * znc * (hit.position.z - center.z) + yz * (hit.position.y - center.y) + zc;


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
		float b = 2.0f*(
		     (ray.start.x - center.x)*ray.dir.x
				+(ray.start.y - center.y)*ray.dir.y
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
		hit.normal.y = 2.0f * (hit.position.y -center.y);
		hit.normal.z = 0.0f ;

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
		float xc = 0.31149, yc =0.130065, zc = 0.880816;

		float a = xnc*powf(dx, 2.0) 
				+ ync*powf(dy, 2.0f) 
				- znc*powf(dz, 2.0f)
				- xy * dx * dy 
				- xz * dx * dz  
				+ yz * dy * dz;
		float b = 2.0f*xnc * dx*(sx - cx) 
				+ 2.0f*ync * dy*(sy - cy) 
				- 2.0f*znc * dz*(sz - cz)  
				- xy* (sx*dy + dx * sy - dx * cy - cx * dy)  
				- xz*(sx*dz + dx * sz - dx * cz - cx * dz)
				+ yz * (sy*dz + dy * sz - dy * cz - cy * dz) 
				- xc*dx
				+ yc*dy  
				+ zc*dz;
		float c = xnc * powf(sx - cx, 2.0f)
				+ ync * powf(sy - cy, 2.0f)
				- znc * powf(sz - cz, 2.0f)
				- xy * (sx*sy - sx * cy - cx * sy + cx * cy) 
				+ yz * (sy*sz - sy * cz - cy * sz + cy * cz)
				- xz * (sx*sz - sx * cz - cx * sz + cx * cz) 
				- xc * (sx - cx)
				+ yc * (sy - cy)  
				+ zc *(sz - cz) - 0.8442548f;

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		
		hit.normal.x = 2.0f * xnc *(hit.position.x - center.x) -   xy * (hit.position.y - center.y) - xz * (hit.position.z - center.z) - xc;
		hit.normal.y = -1.0f* xy * (hit.position.x - center.x) + 2.0f * ync * (hit.position.y - center.y) + yz * (hit.position.z - center.z) + yc;
		hit.normal.z = -1.0f* xz * (hit.position.x - center.x) - 2.0f * znc * (hit.position.z - center.z) + yz * (hit.position.y - center.y) + zc;
		

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
		float a = powf(ray.dir.x, 2.0f) + powf(ray.dir.y, 2.0f) - powf(ray.dir.z, 2.0f);   // forgatás nélküli
		float b = 2.0f*(
			(ray.start.x - center.x)*ray.dir.x
			+ (ray.start.y - center.y)*ray.dir.y
			- (ray.start.z - center.z)*ray.dir.z
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

	Light(vec3 _point, vec3 _Le, float _intensity){
		point = _point;
		Le = _Le;
		intensity = _intensity;
	}
	float getIntensity(float dist) {
		if (dist < 1.0f) return intensity;
		return intensity * (1.0f / (dist * dist));
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		// alapra
		//vec3 eye = vec3(0, 25, 0);vec3 vup = vec3(0, 1, 0);vec3 lookat = vec3(0, 0, 1);
		//vec3 eye = vec3(0, -50, 10);vec3 vup = vec3(0, 1, 0);vec3 lookat = vec3(0, 0, -15);
		//vec3 eye = vec3(30, 15, 8);vec3 vup = vec3(0, 1, 0);vec3 lookat = vec3(0, 0, 0);     //egyszer már működött mind2-re, fentről nézve
		//trafósra
		//vec3 eye = vec3(-25, -5, -10), vup = vec3(0, 1, 0), lookat = vec3(-10, -3, -5);         // transzformáltra

		
		mat4 scaled = ScaleMatrix(vec3(1.0f, 1.0f, 3.0f));
		mat4 rotated = RotationMatrix(0.174532925f,vec3(1.0f,1.0f,3.0f));   // 0.5235987756 rad = 30 deg     0.174532925 = 10 deg
		mat4 translated = TranslateMatrix(vec3(1.0f, 0.0f, 0.0f));
		mat4 srt = scaled*rotated*translated;
		for (int i = 0; i < 4; i++) {
			printf("%f %f %f %f\n", srt.m[i][0], srt.m[i][1], srt.m[i][2], srt.m[i][3]);
		}
		printf("\n\n");


		//lights.push_back(new Light(vec3(180.0f, -70.5f, -10.6f), vec3(2, 2, 2), 30.0f));

		//vec3 eye = vec3(10, -20, -4); vec3 vup = vec3(0, 1, 0); vec3 lookat = vec3(3, 4, 5);  //best3
		//lights.push_back(new Light(vec3(180.0f, -70.5f, -10.6f), vec3(2, 2, 2), 30.0f));

		vec3 eye = vec3(-10,40, 0), vup = vec3(0, 1, 0), lookat = vec3(1, 1, -5); 
		//lights.push_back(new Light(vec3(180.0f, -70.5f, -10.6f), vec3(2, 2, 2), 30.0f));
		
		//vec3 eye = vec3(-2, 20, 2), vup = vec3(0, 1, 0), lookat = vec3(-1,1, 1); // jó mind2-re




		//lights.push_back(new Light(vec3(3.0f, 4.0f, 1.0f), vec3(2, 2, 2), 30.0f));		// simára fasza
		//lights.push_back(new Light(vec3(-10.0f, 3.0f, 1.0f), vec3(2, 2, 2), 30.0f));		// simára fasza
		//lights.push_back(new Light(vec3(2.0f, 1.7f,0.1f), vec3(2, 2, 2), 30.0f));   //oldalról, sima hiperboloidra simára is fasza
		lights.push_back(new Light(vec3(-8.0f, 30.0f, 5.0f), vec3(2, 2, 2), 30.0f));   //oldalról, sima hiperboloidra,


		//lights.push_back(new Light(vec3(10, 0, -30), vec3(2, 2, 2), 30.0f));   //oldalról, sima hiperboloidra,

		//lights.push_back(new Light(vec3(-2.0,10.0, 3), vec3(2, 2, 2), 30.0f));
		//vec3 eye = vec3(-10,50,10), vup = vec3(0, 1, 0), lookat = vec3(0, 1, 0);

		//vec3 eye = vec3(0,30,1), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0); //simára állított nézet
		//lights.push_back(new Light(vec3(8.0f, -8.0, -5.0f), vec3(2, 2, 2), 30.0f));   //tesztelő

		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		Material * material = new Material(kd, ks, 50);

		//objects.push_back(new Sphere(vec3(10.0f, -3.5f, 0.2f),0.2f, material));
		//objects.push_back(new Sphere(vec3(4.6f, 1.8f, 3.2f),0.8f, material));
		//objects.push_back(new Sphere(vec3(8.0f, -13.5f, 13.6f),0.8f, material));    // z minél nagyobb annál balrább,  minusz y minél nagyobb annál lentebb
		//objects.push_back(new Sphere(vec3(14.0f, -4.5f, 1.6f),0.8f, material));    // ez lesz a fasza fény

		//objects.push_back(new Sphere(vec3(-8.0f, 30.0f,5.0f),3.0f, material));    //fény

		//objects.push_back(new Sphere(vec3(2.0f, -8.0f,0.0f),3.0f, material));    

		objects.push_back(new Cylinder(vec3(1.0f, 20.0f,10.0f), 0.0f, material));
		objects.push_back(new Cylinder2(vec3(2.0f, 14.0f,4.0f), 0.0f, material));

		//objects.push_back(new Hyperboloid(vec3(0.0f, 10.0f, 5.0f), 0.0f, material));
		//objects.push_back(new Hyperboloid2(vec3(0.0f, 0.0f, 4.0f), 0.0f, material));
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
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}
	
	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		vec3 sourcePoint; // ahonnan jön a fény
		for (Light * light : lights) {
			sourcePoint  = normalize(light->point - hit.position);
			Ray shadowRay(hit.position + hit.normal * epsilon, sourcePoint);
			float cosTheta = dot(hit.normal, normalize(sourcePoint));
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + sourcePoint);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}
};

// innen nem kell hozzányúlni
GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
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
const char *fragmentSource = R"(
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

FullScreenTexturedQuad * fullScreenTexturedQuad;

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
